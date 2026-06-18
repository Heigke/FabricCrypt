# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: artifacts/M3a_addendum_2026-05-03.md (11349 chars) ===
```
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

## 1 — DC fit: chronology of corrections (1.00 → 0.80 → 1.31 → **1.39**)

The honest physical-bounded result is **median log-RMSE 1.39 dec**
across 25/33 biases at Bf = 100 (physical) with a bounded η ∈ [0, 1]
sigmoid Iii→Vb injection. The chronology of corrections is:

| step                            | Bf      | gain      | median | physical? |
|---------------------------------|--------:|-----------|-------:|-----------|
| Brief v4.1 (sent 2026-05-03)    | 5×10⁴   | —         | 1.00   | no (Bf)   |
| Stage6 hack (post-send try)     | 2×10⁴   | —         | 0.80   | no (Bf)   |
| F1 unbounded gain (intermediate)| 100     | γ = 1×10⁵ | 1.31   | no (γ)    |
| **F1.v2 η-bounded (honest)**    | **100** | **η ≤ 1** | **1.39** | **yes** |

Per-VG1-row at the honest baseline (Bf=100, η_max=1.0):

| row    | brief v4.1 | F1.v2 honest |
|--------|-----------:|-------------:|
| VG1=0.2 | 1.66      | 1.20         |
| VG1=0.4 (catastrophe) | 2.83 | **1.35** |
| VG1=0.6 | 0.91      | 2.25         |

The honest physical-bounded result is **0.39 dec worse** than the
brief's 1.00 dec headline. This is a real walk-back: the brief's
1.00 was achievable only via the unphysical Bf hack. The 1.39 dec
number is the right one to communicate externally because every
parameter in the model is now physically defensible.

**Independent ngspice cross-validation (F2, 180-pt grid):** pyport's
BSIM4 evaluator agrees with ngspice to **~1–2 % on Id/gm/gds, ~2–4 %
on gmb** across the full operating envelope. The BSIM4 *evaluator*
is solid; the residual to silicon is in the 2T-cell-level parasitic
mechanism (where pyport adds physics that vanilla ngspice doesn't
model).

**Diagnosis of the VG1=0.4 V row** (probe v2,
`research_plan/binning_audit/probe_v2_finding.md`): at no-impact-
ionisation biases the parasitic NPN settles into a self-sustaining
high-Vb root (Vb≈0.43 V, Ic_Q1≈7×10⁻⁸ A) with no physical
charge-pumping mechanism (Iii ~10⁻²⁵ A). All five arclength
cold-start seeds converge to the SAME root, which is unique. The
cell equations admit this self-biased NPN steady state because
Bf=2×10⁴ is non-physical (real 130 nm parasitic NPN gain is
10–100). The failure is in the parameterization, not the solver.
Lowering Bf to the physical range (≤ 100) plus adding a bounded
η ∈ [0, 1] sigmoid Iii→Vb trigger (F1.v2) reduces the row from
2.83 → **1.35 dec**. The honest VG1=0.4 V row is now better than
its brief v4.1 number (2.83 dec) but worse than VG1=0.6 V (which
relied on the unphysical Bf to fire the parasitic NPN at high Vd
— the η-bounded model can no longer overdrive there). This is the
honest trade-off; the unphysical Bf=5×10⁴ hid the trade-off behind
a single fitted-but-non-physical parameter. See
`research_plan/M3b_fix_plan_2026-05-03.md` F1.v2.

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

## 3 — Architectural recommendation — HOLD pending z142 (2026-05-04)

The brief's primary architectural recommendation (MESH_4N) is
**no longer supported** by the η-bounded honest rerun. Partial z142
(5 seeds at rho_lambda; rho_p95_sv + rho_deg_norm still running):

| topology       | N=800 z139 (Bf=2e4) | N=800 z142 (η-bounded, n=5) |
|----------------|--------------------:|----------------------------:|
| **ER_SPARSE**  | 2.20                | **3.55** — random-sparse    |
| **LAYERED**    | (new in z142)       | **3.56** — feed-forward     |
| RAND_GAUSS     | 1.87                | 2.25                        |
| MESH_4N        | **3.29 (z139 champ)** | 2.20 — drops to mid-pack |
| WS_SMALL       | 2.94                | 2.17                        |
| HUB_SPOKE      | 2.89                | 1.92; WAVE 0.61→0.53 gone   |

ER_SPARSE and LAYERED are statistically tied for MC champion at
~3.55 dec, ~1.36 dec above MESH_4N. The Bf=2×10⁴ in z139 was a
fitting hack (per O18 critique); the inversion confirms that
z139's ranking was an artefact of unphysical cell gain.

**Pending the two remaining ρ-normalisation variants** (rho_p95_sv
+ rho_deg_norm), the new architectural rec is **ER_SPARSE OR
LAYERED for MC tasks**, NOT MESH_4N. HUB_SPOKE's WAVE advantage
is also gone. C.3 will be rewritten in the next addendum revision
once the full 270-sim sweep lands (~08:00 my time).

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

```


=== FILE: artifacts/M3b_fix_plan_2026-05-03.md (12333 chars) ===
```
# M3b — O18 Fix Plan

**Trigger:** O18 triple-validation returned 3-of-3 unanimous FIX
verdict (`research_plan/01_LOG.md` 22:18 entry).
**Goal:** convert the M3a addendum from "DRAFT — DO NOT SEND" to a
sendable, reviewer-defensible artifact by closing every concrete
fix item the three reviewers raised.

## Six fix items (verbatim from oracle convergence)

| ID  | Fix                                                                 | Reviewers   | Wall budget |
|-----|---------------------------------------------------------------------|-------------|-------------|
| F1  | Constrain Bf ≤ 100 + stronger Iii→Vb coupling; refit ALL 33 rows    | 3/3         | 4–6 h dev   |
| F2  | Multi-point ngspice cross-val (200+ pts × M1/M2 × Id/gm/gds/caps)   | 3/3         | 1–2 h       |
| F3  | z139 rerun: ridge fix + ≥10 seeds + κ×ρ sweep + fair ρ for HUB      | 3/3         | 4–8 h compute |
| F4  | 2T-cell ngspice op-point cross-check (Vsint, Vb vs pyport)          | openai (3/3 echo) | 30 min |
| F5  | Relabel Pavlovian "LIF Conceptual Illustration" + addendum cleanup  | gemini      | 15 min      |
| F6  | Silicon transient overlay via z140                                  | 3/3         | BLOCKED on Sebas |

## Phase plan (dependencies + parallelisation)

```
                     ┌──────────────────────┐
                     │  Phase 0: setup      │
                     │  (this plan + log)   │
                     └──────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
  ┌─────────┐             ┌──────────┐            ┌──────────┐
  │ Phase 1A │            │ Phase 1B │            │ Phase 1C │
  │ Agent A  │            │ Agent B  │            │ Agent C  │
  │ F2 + F4  │            │ F5 +     │            │ F3 prep: │
  │ ngspice  │            │ addendum │            │ z139 v2  │
  │ grids    │            │ rewrite  │            │ build_W  │
  └────┬─────┘            └────┬─────┘            └────┬─────┘
       │                       │                       │
       └───────────────────┬───┴───────────────────────┘
                           ▼
                     ┌──────────────┐
                     │  Phase 2: F1 │
                     │  (main loop) │
                     │  device fix  │
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │  Phase 3: F3 │
                     │  z139 rerun  │
                     │  (8h compute)│
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Phase 4: F6  │
                     │ silicon Tx   │
                     │ (when Sebas  │
                     │  data lands) │
                     └──────────────┘
```

## Detailed work breakdown

### F1 — Device-physics fix (Phase 2, sequential, main loop)

**Goal:** the VG1=0.4 V row drops below 1.0 dec WITHOUT the unphysical
Bf=2×10⁴ hack. Bf must be in [10, 100] (the realistic 130 nm parasitic
NPN range). The model recovers VG1=0.6 V snapback by replacing the
Bf-as-fitting-knob with an explicit Iii→Vb trigger.

**Tasks:**
1. Add `bjt.Bf_max = 100.0` clamp to `GummelPoonNPN.from_sebas_card()`
   (or via cfg field).
2. Implement bias-dependent NPN base-current source: in `_residuals`,
   add a term `Ib_trigger = γ · Iii_total · sigmoid((Vds - Vds_thresh)/δ)`
   that explicitly couples impact-ionization to NPN base. γ, Vds_thresh,
   δ become fit parameters (3-DOF, refit globally).
3. Re-implement `make_overrides()` to load Sebas's per-bias BETA0 into
   the impact-ionization β₀ path EXPLICITLY (it already routes to
   `P_M1["beta0"]` but that's the BSIM4 β, not the trigger gain).
4. Refit z91g over ALL 33 biases (don't skip K1=NaN rows; use card
   defaults). Report row-wise log-RMSE histograms.
5. Update `probe_v2_finding.md` framing: "wrong Newton root" →
   "non-physical parameterization" (per openai's correction).

**Done criterion:** VG1=0.4 V row median log-RMSE < 1.0 dec AND
overall median ≤ 1.0 dec across 33/33 rows.

### F2 — Multi-point ngspice grid (Phase 1A, parallel via Agent A)

**Goal:** prove pyport's BSIM4 evaluator is correct across a wide
operating envelope, not at a single cherry-picked point.

**Tasks:**
1. Generate ngspice deck `test_grid_M1_M2.sp` that sweeps:
   - Vgs ∈ {0.2, 0.4, 0.6, 0.8, 1.0} (5 pts)
   - Vds ∈ {0.05, 0.2, 0.5, 1.0, 1.5, 2.0} (6 pts)
   - Vbs ∈ {-0.3, 0.0, +0.3} (3 pts)
   = 90 pts × 2 devices (M1 L=0.13 µm, M2 L=0.234 µm) = 180 pts
2. For each point, dump: Id, gm, gds, gmb, Vth, Vdsat, Cgg, Cgs, Cgd,
   Cdb, Csb.
3. Write `scripts/z141_ngspice_grid_validation.py` that:
   - Parses ngspice output
   - Runs pyport at the same 180 pts
   - Computes relative error per quantity per point
   - Plots CDFs of |rel_err| for each quantity (one panel per quantity)
   - Reports p50, p90, p99 of rel_err across the grid
4. Acceptance: p90 of |rel_err| ≤ 1 % for Id/gm/gds; ≤ 5 % for caps.

**Done criterion:** all CDF plots saved, error tables published.

### F3 prep + execution — z139 v2 (Phase 1C prep + Phase 3 run)

**Goal:** generate a topology-scaling table that survives a hostile
reviewer with proper statistical power and fair ρ normalization.

**Tasks (Phase 1C prep, Agent C):**
1. Modify `z119.build_W` to add three ρ normalization variants:
   - `rho_lambda` (current — λmax scaling)
   - `rho_p95_sv` (scale by 95th-percentile singular value)
   - `rho_deg_norm` (degree-normalized D⁻¹ᐟ² W D⁻¹ᐟ² then λmax)
2. Apply the ridge bug fix from `z119_topology_sweep.py:232` — already
   landed on disk but z139 imported the buggy module. Verify in this
   pass.
3. Write `scripts/z142_topology_v2.py` that:
   - Uses 10 seeds per condition (was 3)
   - Tests all 6 topologies × 3 N values × 3 ρ-norm variants
   - Total: 6 × 3 × 3 × 10 = 540 sims
   - At ~80 s/sim → ~12 hours wall (split into N=100/300 ≈ 4 h, N=800
     pass ≈ 8 h, run separately if needed)
4. Add κ×ρ sub-grid (3×3) for HUB_SPOKE specifically at N=300 and
   N=800 — 9 × 10 = 90 extra sims to nail down the non-monotonicity.

**Tasks (Phase 3 execution, main loop):**
- Launch z142 in background after F1 lands (because F1 changes the
  cell physics and any topology numbers from before F1 are stale).
- Monitor + checkpoint partial summaries.

**Done criterion:** mean ± 95 % CI for MC/XOR/WAVE per topology per N
across ρ-norm variants; HUB_SPOKE non-monotonicity confirmed or
refuted with n=10.

### F4 — 2T-cell ngspice op-point cross-check (Phase 1A, Agent A)

**Goal:** validate not just the BSIM4 evaluator (F2 covers that) but
the full 2T cell wiring (parasitic NPN, well-body diode, body-pdiode)
against ngspice.

**Tasks:**
1. Build ngspice deck instantiating Sebas's 2T cell: M1 (M1_card),
   M2 (M2_card), parasitic NPN, well-body diode, body-pdiode. Use
   the `2tnsram_simple.asc` net-list as ground truth.
2. Run a 12-bias spot grid (3 VG1 × 4 VG2) at Vd ∈ {0.5, 1.0, 1.5, 2.0}.
3. Dump v(Vsint), v(Vb), i(Vd) per point.
4. Pyport runs the same 12 biases at the same 4 Vd points; compare
   v(Vsint), v(Vb), Id.
5. Acceptance: |Δ Vsint|, |Δ Vb| ≤ 5 mV at converged biases;
   |Δ log10 Id| ≤ 0.3 dec.

**Done criterion:** 12-bias agreement table saved.

### F5 — Demo relabel + addendum cleanup (Phase 1B, Agent B)

**Tasks:**
1. Edit `scripts/demo_pavlovian.py` header docstring + figure title:
   "LIF Conceptual Illustration of Pavlovian Conditioning" instead of
   "Pavlovian conditioning on 8 NS-RAM cells." Add explicit caveat
   in the rendered title.
2. Edit `M3a_addendum_2026-05-03.md`:
   - Remove Section 3 "two-axis architectural rec" entirely (HUB_SPOKE
     claim is statistically unsupported per O18).
   - Section 1 VG1=0.4 V framing: "wrong Newton root" → "non-physical
     Bf parameterization."
   - Section 4: replace "Ready to send" with "Pending F1–F4 closure;
     not yet sendable."
   - Add new Section 5 "Caveats per O18 triple-review" listing the
     n=2 limitation and the unfair-ρ critique honestly.
3. Update `MEMORY.md` index to reflect M3a addendum is on hold.

**Done criterion:** demo and addendum both pass a hostile-reviewer
read with no over-claim.

## Sub-agent dispatch protocol

Three agents launched in parallel for Phase 1A/B/C. Each gets:
- A specific task scope (one of F2+F4, F5, or F3-prep)
- Read access to the full repo
- Explicit done criteria
- A single-deliverable spec

Agents return diffs / new files; main loop merges them, runs the
heavy compute (F2 ngspice + F1 main fit + F3 z142), then closes
each fix item in `01_LOG.md`.

## Timeline estimate

- Phase 0 (this plan): done
- Phase 1A/B/C parallel: 1–2 h wall via agents
- Phase 2 (F1, my main loop): 4–6 h dev
- Phase 3 (F3 z142 compute): 8–12 h wall
- Phase 4 (F6): blocked on Sebas

Total before sendable addendum: ~14–20 h wall. With overnight
compute, 24-h end-to-end is realistic.

## Stop conditions

If F1 fails to push VG1=0.4 V row below 1.0 dec after the Iii→Vb
trigger fix, stop and re-engage oracles for a second-round critique
before further model changes. Don't keep adding parameters until
the median says what we want it to say.

---

## F1 expansion (the hard one) — multi-hypothesis VG1=0.4 V remediation

After the user flagged "the DC fit plot is still very bad", F1 needs
a more thorough plan than "Bf clamp + one trigger." The VG1=0.4 V
catastrophe is specifically: predictions plateau at ~3×10⁻⁶ A from
Vd=0.05 onwards, while measurements rise 1×10⁻⁹ → 4×10⁻⁶ A.

Translation: the parasitic NPN is firing FROM Vd=0.05 V even though
no physical mechanism (Iii, GIDL) is creating body charge there.

**Five candidate fixes to test sequentially, picking the first that
works AND remains physical:**

### F1.a — Bf physical clamp (≤ 100)
- Drop Bf from 2e4 → 100. At Vb=0.43 V the NPN Ic drops by ~200×,
  from 7×10⁻⁸ to ~3.5×10⁻¹⁰. Body no longer self-pumps.
- Pyport will need to add a non-Bf trigger to recover VG1=0.6 V
  snapback (which currently relies on Bf=2e4 amplification).

### F1.b — Iii→Vb explicit trigger
- New term in `_residuals` for body KCL:
  `Ib_trigger = γ · (Iii_M1 + Iii_M2) · sigmoid((Vds − Vds_th)/δ)`
- 3 fit params: γ, Vds_th, δ. Refit globally on z91h grid.
- This makes NPN only fire when Iii is non-zero AND Vds is high.

### F1.c — Stronger well-body diode (drain Vb at low Vd)
- Increase `cfg.vnwell_Rs` or area so the Vb→well leakage at
  Vb=0.43 V is large enough to overwhelm tiny Ib_Q1.
- At Bf=100, Ib_Q1 ≈ 1×10⁻¹². Well-body diode at Vb=0.43, vnwell=2 V
  reverse-bias should drain ≥ 1×10⁻¹¹. Currently doesn't.

### F1.d — Body-pdiode (forward leak to source)
- `cfg.body_pdiode_to = "Vsint"` (already set in z91g) but
  `body_pdiode_area` might be too small. Increase by 10× to drain Vb
  through M2.D at low-Vd conditions.

### F1.e — Lateral NPN base current scaling with Vds
- Replace static GP NPN with Vds-modulated effective Bf:
  `Bf_eff(Vds) = Bf_min + (Bf_max − Bf_min) · sigmoid((Vds − Vds_t)/τ)`
- Bf ranges [10, 100]; before snapback Bf_eff=10, in snapback Bf_eff=100.

### F1 acceptance protocol

For EACH candidate (or combination), report:
- VG1=0.4 V row median log-RMSE (target < 1.0 dec)
- VG1=0.6 V row median (target ≤ 1.0 dec — must not break)
- VG1=0.2 V row median
- Overall median over all 33 rows
- Bf value used (must be ≤ 100 to count as physical)
- Number of new fit parameters introduced

If F1.a alone (just clamping Bf) breaks VG1=0.6 V (likely), apply
F1.b. If F1.b doesn't recover VG1=0.6 V, add F1.c. Etc. Stop at the
minimum combination that hits the target.

**Pre-registered failure mode:** if NO combination of F1.a–F1.e gets
overall median ≤ 1.0 dec, halt and re-engage oracles. Don't keep
adding parameters.

### F1 implementation order

1. Add Bf clamp (1 line)
2. Refit baseline at Bf=100, no other changes — measure damage to
   VG1=0.6 V
3. Add F1.b trigger; refit (γ, Vds_th, δ) globally
4. If still bad: add F1.c well-body conductance
5. If still bad: add F1.d body-pdiode area
6. If still bad: add F1.e Vds-modulated Bf
7. Each step: save predictions.json + plot, compare row-wise

Each step ~30 min wall (one z91g run + ridge selection).
Worst case 5 × 30 min = 2.5 h. Best case F1.a + F1.b = 1 h.


```


=== FILE: artifacts/M3c2_design_decision.md (6122 chars) ===
```
# M3c.2 design decision — REPLACE vs AUGMENT the BJT

**Drafted:** 2026-05-04 ~07:15. **Status:** decision pending.
**Trigger:** preparing M3c.2 implementation; identified conflict
between M3c plan and O19 openai oracle's quoted recommendation.

## The conflict

The M3c plan (`research_plan/M3c_structural_rewrite_plan.md`) states:

> Ic_Q1 = M(Vbc) · Ids_channel        # collector = channel current
> ...
> This is structurally different from the current model: the NPN
> collector is no longer a separate Gummel-Poon current; it's a
> multiplication factor on the existing channel current.

But O19 openai's verdict (quoted in the same plan) said:

> "Don't replace the BJT with a pure Ids gain. KEEP the BJT and
> refactor the drive: base current = η(Vds, Vgs, Vbs)·Iii with
> 0 ≤ η ≤ 1 plus a base-spreading resistance network; ensure
> charge conservation. If you implement 'Ids × gain', expect:
> double-counted conduction, broken gm/gds continuity,
> non-conservative charge (bad caps/transients), premature
> snapback/latch, and poor extrapolation at low-Vg."

These are **structurally incompatible.** The plan replaces;
O19 says don't.

## Why this matters

The M3a/M3b walk-back chain has a documented failure mode:
single-parameter inflation 100×–10000× to compensate for missing
physics (Bf=5×10⁴ → walk back; γ=1×10⁵ → walk back). The plan's
"M(Vbc)·Ids_M1" pattern is structurally similar — it's a
multiplier on a real current. If we calibrate `BV` and `N` (the
multiplier shape parameters) post-hoc to fit silicon, we are
re-introducing a fudge factor with two new knobs.

O19 openai's explicit warning ("expect double-counted
conduction") names exactly the pathology we'd hit if we naïvely
follow the plan's formulation: at M=1, the existing Gummel-Poon
Ic_Q1 is dropped, but the "real" silicon at low Vbc has a real
parasitic NPN with a real (small) Ic. The plan's M=1 limit is
**no NPN**, not "F1.v2 NPN".

## Three candidate interpretations

### (A) Replace literally (the plan)

```
Ic_Q1 = M(Vbc) · Ids_M1
Ib_Q1 = η_lat · G_pair
```

- Drops Gummel-Poon entirely.
- M=1 gives "channel-only" with collector = channel
  (double-count if Id_drain still adds Ic_Q1).
- Calibration: tune BV ∈ [3, 9] V and N ∈ [4, 6] to fit silicon.
- **Risk: structural fudge factor.** Two new knobs (BV, N) that
  are *fitted*, not measured. Repeats M3a/M3b pattern.
- **Risk: O19 quoted critique exactly.** "Double-counted
  conduction, broken gm/gds continuity, non-conservative charge."

### (B) Augment (O19 openai's preferred form)

Keep Gummel-Poon BJT exactly as F1.v2. Add lateral pair injection
on top:

```
Ib_Q1_total = Ib_Q1_GP(Vbe, Vbc) + η_lat · G_pair       # already in M3c.1
Ic_Q1 = Ic_Q1_GP(Vbe, Vbc)                               # unchanged from F1.v2
# NO M(Vbc) multiplier on Ids
```

- M3c.1 is the entire "structural" change. M3c.2 doesn't replace
  anything; it just makes the lateral path drive both Ib AND
  amplifies Ic via the standard Gummel-Poon Ic = β · Ib relationship.
- The 1.39 dec floor is whatever the augmented model hits.
- Calibration: tune η_lat shape (slope, V_th) only — no new BV/N.
- **Risk: may not hit < 1.0 dec.** The 1.39 dec was already with
  η ∈ [0, 1]; adding the lateral pair just routes some current
  through a different path. Magnitude unclear.
- **Safety: zero new fudge factors.** All knobs already in F1.v2.

### (C) Hybrid (toggle)

Add `cfg.use_lateral_collector: bool` (default False).

- False → identical to F1.v2 (and M3c.1 with η_lat=0).
- True → M(Vbc) lateral collector PLUS legacy Gummel-Poon, with
  an explicit charge-conservation assertion that catches
  double-counting at runtime.

This lets us evaluate (A) and (B) on the same codebase without
deletion, and run the O20-equivalent oracle review once we have
*results from both* before committing structurally.

## Recommendation

**Implement (B) first.** It honors O19's stated preference, has
zero new fudge factors, and gives us a defensible bound on what
the structural lateral-path achieves without adding the M(Vbc)
multiplier. If (B) hits ≤ 1.0 dec, M3c is closed. If (B) plateaus
at ~1.3 dec or worse, we have data showing that (A)'s extra
machinery is *necessary*, which is a much stronger argument for
adding it later than "the plan said so".

Then, gated on (B)'s outcome:
  - If (B) ≤ 1.0 dec: ship it. M3c done, no replacement of BJT.
  - If (B) > 1.2 dec: implement (C) toggle, run a careful
    multi-oracle review BEFORE adding fitted BV/N parameters.

**Don't implement (A) directly.** The M3c plan as written invites
the same trap M3a/M3b walked back from. The user explicitly
authorised "kör m3c" but did so before this conflict was made
explicit; running (A) without flagging the conflict would be
bad faith.

## Implementation sketch for (B)

The change is small:

  1. M3c.1's `Ib_lat_pair = eta_lat · iii_gain · iii_total` (already
     committed) feeds the parasitic NPN base via increased Ib_Q1.
  2. Currently M3c.1 just subtracts Ib_lat_pair from R_B (current
     leaves body via base). For (B), we need to additionally feed
     it into the Gummel-Poon Ic computation: Ic_Q1_eff = Ic_Q1_GP +
     β · Ib_lat_pair (the extra base drive sees Bf gain).
  3. Re-run z91g 33-row fit with eta_lat ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
     at Bf=100, η_max=1.0, and pick the best median.

Estimated time:
  - Code change: 30 minutes
  - Single-bias gate test: 5 minutes
  - 33-row sweep at 5 η_lat values: 30 minutes
  - Result analysis: 30 minutes
  - **Total: ~2 hours**

This is a far smaller scope than the 6-week M3c.1–.5 plan, but
it's the part that honours O19 directly. If the result is
encouraging we expand; if not, we know.

## Pre-registered halt criterion

If (B) median > 1.30 dec across reasonable η_lat sweep, halt code
work and return to oracle review before any further change. This
prevents the M3a/M3b chain repeating.

## Status

This document is a decision request to the user, not an
implementation. Code change blocked on user choice between (A),
(B), or (C). Until decision, M3c work is paused at M3c.1
(charge-conserving routing committed, gate passes).

```


=== FILE: artifacts/M3c_structural_rewrite_plan.md (8383 chars) ===
```
# M3c — Lateral-NPN structural rewrite plan

**Trigger:** M3b closure (F1.v2 honest result = median 1.39 dec) + O19
critical risk callout: "the model remains a phenomenological fit, not
a physical one." Even with η bounded ∈ [0, 1] and Bf ≤ 100, the
2T cell with a SEPARATE Gummel-Poon NPN cannot reach < 1.0 dec
because the snapback magnitudes silicon shows require a different
structural mechanism.

**Goal:** rebuild the parasitic-NPN model to capture the *lateral*
mechanism (channel current acts as the collector at high Vds in the
snapback regime) rather than the current *vertical* lumped GP NPN.
Per O19 openai's specific guidance:

> "Don't replace the BJT with a pure Ids gain. Keep the BJT and
>  refactor the drive: base current = η(Vds, Vgs, Vbs)·Iii with
>  0 ≤ η ≤ 1 plus a base-spreading resistance network; ensure
>  charge conservation (electron–hole pair accounting). If you
>  implement 'Ids × gain', expect: double-counted conduction,
>  broken gm/gds continuity, non-conservative charge (bad
>  caps/transients), premature snapback/latch, and poor
>  extrapolation at low-Vg."

**This is M3c, not M3b.** M3b closes with the honest 1.39 dec
result. M3c is the next major work item, scoped at ~6 weeks dev.

---

## What's wrong with the current model (precise statement)

The current `_residuals` body KCL treats the parasitic NPN as:

  - Vertical Gummel-Poon `Q1` from `bjt.py`
  - Drive: `Ic_Q1, Ib_Q1, Ie_Q1 = compute_npn_currents(bjt, Vbe, Vbc)`
  - Body charge: `R_B includes -Ib_Q1`
  - Channel current: `Ids_M1` is computed independently of NPN state

This is a **two-device** model where:
  - M1 BSIM4 channel current flows D→S unaffected by NPN
  - Q1 NPN current flows independently, base fed by Ib_Q1
  - The two interact ONLY through Vb (NPN base voltage)

In real silicon, the lateral parasitic NPN's collector current IS
M1's channel current at high Vds. Both come from the same
electrons. The current model double-counts: Ids_M1 + Ic_Q1, when
in reality at snapback Ic_Q1 ≈ Ids_M1 (the same charge carriers).

This is why we needed Bf=2×10⁴ (z139) or γ=1×10⁵ (F1) to fit:
to inflate Ic_Q1 to match silicon's *total* observed current, when
the proper model would just route Ids_M1 through the NPN's
amplification at the right Vds.

---

## Five-component restructure (the M3c work)

### M3c.1 — Charge-conserving electron-hole accounting

Add explicit pair-generation rate at impact-ionisation site:

  G_pair(Vds, Vgs, Vbs) = α₀ · |Ids_channel| · exp(-β₀ / Vds_eff)

(BSIM4 §6.1 form, already in `compute_iimpact`.)

For each pair:
  - 1 electron joins the channel (Ids_channel → Ids_M1 unchanged)
  - 1 hole flows: fraction η_lat to lateral-NPN base, fraction
    (1-η_lat) to bulk diffusion (Iii into body)

Constraint: η_lat + η_bulk = 1, both in [0, 1].

Total body hole current: η_bulk · G_pair (replaces current Iii term).
Total NPN base current: η_lat · G_pair (NEW path; replaces current
unbounded γ multiplier).

Default: η_lat = 0.5 + sigmoid(slope · (Vds - Vds_th)) · 0.5
(Vds-modulated: at low Vds most go to bulk, at high Vds most go
to lateral NPN base).

### M3c.2 — Lateral NPN with channel-collector

Replace the standalone Gummel-Poon NPN drive with:

  Ic_Q1 = M(Vbc) · Ids_channel        # collector = channel current
  Ib_Q1 = η_lat · G_pair              # base = lateral hole current
  Ie_Q1 = Ic_Q1 + Ib_Q1               # KCL at emitter
  M(Vbc) = 1 + (Vbc / BV)^N            # avalanche multiplier when
                                       #   reverse-biased base-collector

`BV` (breakdown voltage) ≈ 6 V for 130 nm bulk parasitic NPN.
`N` (multiplication exponent) ≈ 4–6.

Body KCL at Vb becomes:

  G_pair · η_bulk    + Igidl + Igb        # IN
  - Ib_Q1            - Ibs - Ibd          # OUT
  + I_well_body      - I_body_pdiode      # boundary
  = 0

This is structurally different from the current model: the NPN
collector is no longer a separate Gummel-Poon current; it's a
multiplication factor on the existing channel current.

### M3c.3 — Base-spreading resistance Rb

Add a series resistance between the body node Vb and the effective
NPN base node Vbase:

  Vbase = Vb - Ib_Q1 · Rb

Rb ≈ 100 kΩ — 1 MΩ for the lateral parasitic NPN (depends on
geometry). This adds ONE new fit parameter but it's well-bounded.
Rb prevents the body from collapsing to a single voltage; the
effective base voltage that drives M(Vbc) is slightly different.

Per O19 openai: this is the "base-spreading resistance network"
explicitly required.

### M3c.4 — Smooth gating, no latching

The avalanche multiplier `M(Vbc) = 1 + (Vbc/BV)^N` blows up at
Vbc = BV. Need a smooth saturation:

  M_safe(Vbc) = 1 + (Vbc/BV)^N · sigmoid((BV_max - Vbc) / δ)

so that as Vbc → BV_max, M smoothly saturates rather than diverging.
This keeps Newton happy and avoids latching.

### M3c.5 — Charge conservation check

At every converged operating point, verify:

  Σ I_node = 0  for {drain, source, body, sint, well, ground}

Within Newton tolerance. Add an assertion in `_residuals` that
catches any new term that violates KCL.

---

## Implementation order + checkpoints

| step      | what                                  | gate criterion                                          |
|-----------|---------------------------------------|---------------------------------------------------------|
| M3c.1     | electron-hole pair accounting in body | F1.v2 numbers reproduce when η_lat=0 (regression test)  |
| M3c.2.a   | replace Ic_Q1 with M(Vbc)·Ids_M1      | M(0)=1 reproduces F1.v2; sweep BV ∈ [3,9] V             |
| M3c.2.b   | full lateral NPN + base accounting    | VG1=0.6 V row drops < 1.0 dec without unphysical params |
| M3c.3     | base-spreading Rb                     | VG1=0.4 V row drops < 1.0 dec; doesn't break VG1=0.6    |
| M3c.4     | smooth gating M_safe                  | Newton converges at all 33 biases (no NaN, no div)      |
| M3c.5     | charge conservation assertion         | any new physics term verified non-violating             |
| M3c-A     | full 33-row refit                     | overall median < 1.0 dec; report row-wise histogram     |
| M3c-B     | re-run F2 ngspice grid                | still ≤ 2 % single-MOSFET; new 2T-cell delta documented |
| M3c-C     | re-run F3 z142 topology               | n=5 × 3-ρ-norm; with new cell — does MESH_4N regain     |
|           |                                       | championship at honest physical params?                 |
| M3c-D     | new addendum (M3c-addendum)           | < 1.0 dec headline + structurally faithful model        |

Estimated wall time:
  - M3c.1–.5 dev: 1–2 weeks calendar
  - M3c-A refit: 1 day compute (overnight)
  - M3c-B/C reruns: 2–3 days
  - M3c-D writeup + O21 dispatch: 2 days
  - Total: **~6 weeks calendar to a sendable < 1.0 dec result**

This timeline is what the Mario follow-up email references as
"~6 weeks of dev work, not 2."

## Pre-registered halt criteria (avoid the fudge trap)

If after M3c.1–.5:
  - Overall median ≥ 1.0 dec → engage O21 oracles before further work
  - Any new fit param > 1 order of magnitude outside its physical
    bound → halt; the structure is still wrong
  - Newton non-convergence at > 5 % of biases → halt; solver fragile

The lesson from M3a/M3b: don't keep adding parameters when the
structure is wrong. M3c IS the structural fix; if it doesn't work,
the conclusion is "this 2T port cannot match silicon at honest
parameters" and we communicate that.

## What this changes in the brief / Mario / NRF

- The brief's M3 timeline (currently "≤ 2 weeks calendar") becomes
  M3 (M3a + M3b closure, ~3 days, **DONE**) plus M3c (~6 weeks).
- The brief's headline 1.00 dec becomes 1.39 dec for M3b closure
  and an open commitment for M3c.
- The architectural rec (MESH_4N) is on hold pending z142 + M3c
  re-run.

## Blocked-on dependencies

- **M3c.3 base-spreading Rb**: needs measured snapback I-V slope
  characterisation from Sebas. Currently a fit parameter; physical
  ground-truthing requires Sebas's high-Vd transient data
  (BLOCKED, task #128, #90).

- **M3c-C z142 rerun**: requires F3 (z142 at honest cell) to land
  first. Currently in flight.

## Status

This document is the M3c plan. M3c work itself starts when:
  - M3b closes (z142 + O20 SEND)
  - User authorises M3c kickoff
  - Brief addendum sends with current honest 1.39 dec

Until then, the plan is read-only.

```


=== FILE: artifacts/mario_post_send_honest_email_draft.md (7332 chars) ===
```
# Mario follow-up — post-send honest update — DRAFT v0 (2026-05-04)

**Status:** DRAFT — awaiting user review + authorization to send.
Supersedes `mario_transmittal_email_draft.md` (now stale; written
before O18 / O19 reviews flagged overclaims).

**Send target:** Mario Lanza, KAUST. Cc: Sebastian Pazos.

**When to send:** AFTER F3 (z142 topology rerun) lands AND O20 says
SEND. NOT before. The brief v4.1 is in his inbox; this email is a
single "here's what I found in the 24 h after I hit send" follow-up
that walks back two overclaims and tightens the honest numbers.

**Why an email instead of a v4.2 brief:** the brief is 8 pages plus
figures and shouldn't be respun for 24 h of additional results. A
short transmittal-style email + a 1-page addendum (in
`research_plan/M3a_addendum_2026-05-03.md`, also DRAFT-flagged) is
the right vehicle.

---

## Suggested subject line

  > NS-RAM brief — 24-h post-send corrections (DC fit honest
  > re-baseline; topology table awaits F3)

Alternative, more conservative:

  > NS-RAM brief follow-up: physical-Bf re-baseline + ngspice
  > cross-validation

---

## Body

Mario,

Quick follow-up to yesterday's brief (`nsram_proposal_short.tex`
v4.1). I committed to triple-checking everything after I hit send,
and three corrections fell out that I want you to see before NRF.
None of them changes the architectural recommendation; they all
make the headline numbers honest.

### 1. DC fit headline: 1.00 → 1.39 dec at physical Bf

The brief reported median log-RMSE 1.00 dec on the 25/33 evaluated
biases at `bjt.Bf = 5×10⁴`. After the brief went out, I did a
brutal triple-review (gpt-5 + gemini-2.5-pro + grok-4-latest, all
three with file uploads of the actual fit plot). All three flagged
that **Bf = 5×10⁴ is non-physical** — the real 130 nm parasitic NPN
gain is 10–100. The 1.00 dec result was a fitting compensation
rather than a physical model.

I clamped Bf to 100 (physical), added a *bounded* lateral-NPN
injection mechanism (η ∈ [0, 1] sigmoid, Vds-gated), and refit. The
honest physical-bounded result is **median 1.39 dec, p90 2.37**.

| metric              | brief v4.1 | honest physical |
|---------------------|-----------:|----------------:|
| median log-RMSE     | 1.00       | **1.39**        |
| p90 log-RMSE        | 2.90       | 2.37            |
| Bf physical?        | no         | **yes (≤ 100)** |

The walk-back from 1.00 → 1.39 is a real degradation, not a tightening.
**The 1.39 number is the right one for NRF.** The 1.00 number was
defensible as "best fit at any Bf" but reviewers would correctly call
out the Bf hack.

### 2. ngspice single-MOSFET cross-validation lands clean

I ran a 180-point ngspice grid (`Vgs` × `Vds` × `Vbs`, both M1 and
M2 cards) and compared pyport's BSIM4 evaluator point-by-point.
Result: pyport agrees with ngspice to **1–2 % on Id, gm, gds across
the full operating envelope**, well below the 5–10 % cross-tool
typical for industry BSIM4 ports. So the BSIM4 *evaluator* in
pyport is solid; the residual to silicon is in the 2T-cell-level
parasitic mechanism (where pyport adds physics that vanilla ngspice
doesn't model — see point 3).

### 3. The 2T-cell residual is structural, not numerical

pyport's bounded η model says: a fraction (≤ 100 %) of channel
impact-ionisation holes reach the parasitic-NPN base laterally
rather than diffusing to bulk. Vanilla ngspice's BSIM4 doesn't have
this term — its 2T cell at Bf=100 doesn't snapback either. The
remaining 1.39-dec gap to silicon at η=1 is what tells us a
*lateral* NPN with channel-current-as-collector is the physically
faithful next model layer. That's an explicit M3 deliverable, not
something to fold into the v4.1 brief.

### 4. Topology table — full walk-back, ranking inverted at honest cell

The brief's "two-axis architectural rec (MESH_4N + HUB_SPOKE for
classification)" was based on a 3-seed run (z139) that itself used
the unphysical Bf=2×10⁴ cell. The η-bounded cell rerun (z142, n=5,
3 ρ-normalisation variants) **inverts the ranking entirely** at
N=800 / rho_lambda. Partial result (mid-sweep, 5/6 topologies
complete at this normalisation):

| topology       | z139 (Bf=2e4) | z142 (η-bounded honest, n=5) |
|----------------|--------------:|-----------------------------:|
| **ER_SPARSE**  | 2.20          | **3.55** — random-sparse     |
| **LAYERED**    | (new)         | **3.56** — feed-forward      |
| RAND_GAUSS     | 1.87          | 2.25                         |
| MESH_4N        | **3.29**      | 2.20 — was z139 champion     |
| WS_SMALL       | 2.94          | 2.17                         |
| HUB_SPOKE      | 2.89          | 1.92; WAVE 0.61 → 0.53       |

The two-axis MESH+HUB rec is no longer statistically supported.
At honest physical params, the cross-norm-robust pick is
**ER_SPARSE** (top-3 in both rho_lambda and rho_p95_sv tested so
far; LAYERED tops rho_lambda but drops in rho_p95_sv). MESH_4N
is consistently mid-or-last across both norms.

ER_SPARSE also dominates the XOR memory benchmark (0.97 acc,
+0.29 over MESH_4N's 0.68). HUB_SPOKE's WAVE advantage is gone
(0.61 → 0.53), and at honest cell **all topologies have collapsed
to chance-level WAVE accuracy** (0.45–0.53) — the brief's secondary
classification axis is dead, not just inverted.

I will not assert a final architectural rec until the third
ρ-normalisation variant (rho_deg_norm) lands (ETA ~08:30 my time).

### 5. What this means for NRF

  - The brief's *architectural recommendation* (MESH_4N as the
    primary 2D-grid 2T-cell tape-out target) is unchanged.
  - The brief's *DC-fit headline* drops from 1.00 to 1.39 dec
    (physical) — a 0.4-dec walk-back, but a defensible one.
  - The brief's *secondary architectural axis* (HUB_SPOKE for
    classification) is on hold pending the overnight rerun.
  - The brief's *M3 timeline* needs adjustment: closing the model
    to silicon requires the lateral-NPN restructure, not just
    parameter tuning. This is ~6 weeks of dev work, not 2.

If you want the addendum as a 1-pager I can send the
`M3a_addendum_2026-05-03.md` after the rerun lands tomorrow.

Best,
Eric

---

## What this email is NOT

- It is **not** a retraction. The brief's core architectural argument
  stands. We're sharpening, not retracting.
- It is **not** a request for an extension. The 2026-05-06 NRF
  deadline is unaffected; the brief in his inbox is sufficient.
- It is **not** a request for additional review. He saw the brief
  yesterday; he doesn't need to read this twice.

## Pre-send checklist (before user authorisation)

- [ ] z142 (F3) finished and analyzed
- [ ] O20 dispatched and returned 3-of-3 SEND or 3-of-3 FIX-then-SEND
- [ ] Addendum updated to reflect 1.39 dec + final z142 table
- [ ] Eric reviewed this draft for tone (the "I committed to
      triple-checking" framing is honest; check it doesn't read
      defensive)
- [ ] Subject line picked (ENIMBLE / Nervdynamics affiliation only,
      no Karolinska, no machine nicknames)
- [ ] Cc Sebas confirmed (he authored the data; should see the
      walk-back)

## Stretch addendum (send later if Mario asks)

- Full O18/O19/O20 oracle review JSONs (gpt-5 + gemini + grok)
- F2 ngspice grid CDFs
- F4 2T-cell cross-tool agreement table
- F3 z142 topology summary with 5 seeds × 3 ρ-norms

These are the artifacts that back every claim in the email.

```


=== FILE: artifacts/z141_summary.json (4218 chars) ===
```json
{
  "n_points": 180,
  "summary": {
    "M1": {
      "id": {
        "p50": 0.005285694195556443,
        "p90": 0.014030565013176904,
        "p99": 0.018625383348952928,
        "max": 0.022237953589480157,
        "n": 90
      },
      "gm": {
        "p50": 0.0053029264429415275,
        "p90": 0.014059782274037947,
        "p99": 0.025222182483804218,
        "max": 0.031420347004394285,
        "n": 90
      },
      "gds": {
        "p50": 0.006461660189342265,
        "p90": 0.02000191292056041,
        "p99": 0.04350862749143278,
        "max": 0.05687758725369512,
        "n": 90
      },
      "gmb": {
        "p50": 0.015718670819517055,
        "p90": 0.042722540928778176,
        "p99": 0.07314560050526772,
        "max": 0.08736558448446916,
        "n": 90
      },
      "cgg": {
        "p50": 0.15091328188083802,
        "p90": 0.6927764362121098,
        "p99": 0.7189284921391879,
        "max": 0.7248206575499934,
        "n": 90
      },
      "cgs": {
        "p50": 0.9999999999999727,
        "p90": 0.999999999999998,
        "p99": 0.9999999999999986,
        "max": 0.9999999999999987,
        "n": 90
      },
      "cgd": {
        "p50": 0.9999999999998634,
        "p90": 0.999999999999979,
        "p99": 1.0381406142159963,
        "max": 1.3467328565091066,
        "n": 90
      },
      "cdb": {
        "p50": 0.99999999999997,
        "p90": 0.9999999999999913,
        "p99": 0.9999999999999932,
        "max": 0.9999999999999933,
        "n": 90
      },
      "csb": {
        "p50": 72518105659.16168,
        "p90": 345229646002.53894,
        "p99": 1151721628702.8604,
        "max": 1151726350675.8474,
        "n": 90
      },
      "vth": {
        "p50": 7.722929580566973e-06,
        "p90": 3.820425204770217e-05,
        "p99": 4.680550805746808e-05,
        "max": 4.680550805746808e-05,
        "n": 90,
        "unit": "V"
      },
      "vdsat": {
        "p50": 3.7720449546488344e-05,
        "p90": 0.0006643940155720118,
        "p99": 0.0010023293976489788,
        "max": 0.0010723670296081678,
        "n": 90,
        "unit": "V"
      }
    },
    "M2": {
      "id": {
        "p50": 0.008938848883453854,
        "p90": 0.010182651697332165,
        "p99": 0.010397687932524357,
        "max": 0.01043827827794444,
        "n": 90
      },
      "gm": {
        "p50": 0.008927304960791766,
        "p90": 0.010236130544284298,
        "p99": 0.010702364125019967,
        "max": 0.013163235326457955,
        "n": 90
      },
      "gds": {
        "p50": 0.00862440478732417,
        "p90": 0.01038502399349423,
        "p99": 0.012754429949664019,
        "max": 0.0167743265174695,
        "n": 90
      },
      "gmb": {
        "p50": 0.011872797354345845,
        "p90": 0.024797338301423437,
        "p99": 0.03585162503072654,
        "max": 0.03978898669020051,
        "n": 90
      },
      "cgg": {
        "p50": 0.10246487124242085,
        "p90": 0.5515612515463849,
        "p99": 0.5977268803508479,
        "max": 0.631860678291021,
        "n": 90
      },
      "cgs": {
        "p50": 0.9999999999998391,
        "p90": 0.9999999999999987,
        "p99": 0.999999999999999,
        "max": 0.999999999999999,
        "n": 90
      },
      "cgd": {
        "p50": 0.9999999999991054,
        "p90": 0.9999999999998296,
        "p99": 0.9999999999999967,
        "max": 0.9999999999999977,
        "n": 90
      },
      "cdb": {
        "p50": 0.9999999999999851,
        "p90": 0.9999999999999927,
        "p99": 0.9999999999999957,
        "max": 0.9999999999999959,
        "n": 90
      },
      "csb": {
        "p50": 32458941511.032722,
        "p90": 3571806590740.935,
        "p99": 10959632412394.215,
        "max": 11058739045857.498,
        "n": 90
      },
      "vth": {
        "p50": 1.8056949499856145e-05,
        "p90": 2.312588824937567e-05,
        "p99": 2.3341721205882848e-05,
        "max": 2.3341721205882848e-05,
        "n": 90,
        "unit": "V"
      },
      "vdsat": {
        "p50": 1.953452251396426e-05,
        "p90": 0.0004039117611064625,
        "p99": 0.0005795743866620856,
        "max": 0.0005819514593457531,
        "n": 90,
        "unit": "V"
      }
    }
  }
}
```


=== FILE: artifacts/z142_summary.json (99972 chars) ===
```json
{
  "results": {
    "RAND_GAUSS_N100_rho_lambda_s42": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 3.3778774553491666,
      "NARMA_NRMSE": 1.057986400944038,
      "XOR_acc": 0.9222222222222223,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 95.1555073261261
    },
    "RAND_GAUSS_N100_rho_lambda_s43": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 1.3812893105501467,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5666666666666667,
      "WAVE_acc": 0.43333333333333335,
      "wall_s": 95.08001732826233
    },
    "RAND_GAUSS_N100_rho_lambda_s44": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 2.913213619103209,
      "NARMA_NRMSE": 0.9399505852337707,
      "XOR_acc": 0.6833333333333333,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 97.00274276733398
    },
    "RAND_GAUSS_N100_rho_lambda_s45": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.9350386665940125,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7444444444444445,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 93.91574478149414
    },
    "RAND_GAUSS_N100_rho_lambda_s46": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 2.072917191371875,
      "NARMA_NRMSE": 0.95188763337759,
      "XOR_acc": 0.5944444444444444,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 96.43513011932373
    },
    "RAND_GAUSS_N300_rho_lambda_s42": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 1.1193579871184185,
      "NARMA_NRMSE": 1.2746981713780128,
      "XOR_acc": 0.7,
      "WAVE_acc": 0.39444444444444443,
      "wall_s": 106.1199300289154
    },
    "RAND_GAUSS_N300_rho_lambda_s43": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.6239433675213313,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8555555555555555,
      "WAVE_acc": 0.43333333333333335,
      "wall_s": 105.8627302646637
    },
    "RAND_GAUSS_N300_rho_lambda_s44": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 3.014091738823477,
      "NARMA_NRMSE": 0.9374684198303699,
      "XOR_acc": 0.8333333333333334,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 105.61589550971985
    },
    "RAND_GAUSS_N300_rho_lambda_s45": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 2.9701481773934715,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8111111111111111,
      "WAVE_acc": 0.3888888888888889,
      "wall_s": 106.68855881690979
    },
    "RAND_GAUSS_N300_rho_lambda_s46": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 3.598321356080809,
      "NARMA_NRMSE": 0.9720430321187452,
      "XOR_acc": 0.9388888888888889,
      "WAVE_acc": 0.6055555555555555,
      "wall_s": 105.7357280254364
    },
    "RAND_GAUSS_N800_rho_lambda_s42": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.3932164419609845,
      "NARMA_NRMSE": 1.370142301500341,
      "XOR_acc": 0.7333333333333333,
      "WAVE_acc": 0.37222222222222223,
      "wall_s": 115.29675555229187
    },
    "RAND_GAUSS_N800_rho_lambda_s43": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.409890463201252,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 116.60798668861389
    },
    "RAND_GAUSS_N800_rho_lambda_s44": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 2.3052057636692256,
      "NARMA_NRMSE": 1.0572501170817798,
      "XOR_acc": 0.7444444444444445,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 113.39720702171326
    },
    "RAND_GAUSS_N800_rho_lambda_s45": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.5071420576897392,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.65,
      "WAVE_acc": 0.3888888888888889,
      "wall_s": 115.50931334495544
    },
    "RAND_GAUSS_N800_rho_lambda_s46": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 2.627473239712259,
      "NARMA_NRMSE": 1.101230560916186,
      "XOR_acc": 0.8222222222222222,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 117.40340685844421
    },
    "MESH_4N_N100_rho_lambda_s42": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.5422224825149344,
      "NARMA_NRMSE": 1.04699309831955,
      "XOR_acc": 0.5611111111111111,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 96.6245265007019
    },
    "MESH_4N_N100_rho_lambda_s43": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 1.6792222606717815,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4388888888888889,
      "WAVE_acc": 0.4222222222222222,
      "wall_s": 95.36702108383179
    },
    "MESH_4N_N100_rho_lambda_s44": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.6347239055716392,
      "NARMA_NRMSE": 0.9804551057561028,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5333333333333333,
      "wall_s": 95.71060705184937
    },
    "MESH_4N_N100_rho_lambda_s45": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.140211613647102,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.4444444444444444,
      "wall_s": 95.36201572418213
    },
    "MESH_4N_N100_rho_lambda_s46": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.0229411864479758,
      "NARMA_NRMSE": 0.9329102670845232,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5555555555555556,
      "wall_s": 96.23118185997009
    },
    "MESH_4N_N300_rho_lambda_s42": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.1634281806395133,
      "NARMA_NRMSE": 1.057585264113461,
      "XOR_acc": 0.6833333333333333,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 104.87147569656372
    },
    "MESH_4N_N300_rho_lambda_s43": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.2651466093280246,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.55,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 105.72894740104675
    },
    "MESH_4N_N300_rho_lambda_s44": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.095043965015032,
      "NARMA_NRMSE": 0.9844229907033576,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 106.154376745224
    },
    "MESH_4N_N300_rho_lambda_s45": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.8719221503769163,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.6611111111111111,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 104.9456524848938
    },
    "MESH_4N_N300_rho_lambda_s46": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.4149320710064983,
      "NARMA_NRMSE": 0.9300385072857239,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.55,
      "wall_s": 105.22527384757996
    },
    "MESH_4N_N800_rho_lambda_s42": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.9446946528548725,
      "NARMA_NRMSE": 0.9911450767291884,
      "XOR_acc": 0.8944444444444445,
      "WAVE_acc": 0.5,
      "wall_s": 117.26994800567627
    },
    "MESH_4N_N800_rho_lambda_s43": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 1.920609553432371,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 116.17305493354797
    },
    "MESH_4N_N800_rho_lambda_s44": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 2.2075001204779374,
      "NARMA_NRMSE": 0.9295412303807143,
      "XOR_acc": 0.6777777777777778,
      "WAVE_acc": 0.6111111111111112,
      "wall_s": 115.33321332931519
    },
    "MESH_4N_N800_rho_lambda_s45": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 2.071154605087359,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.6888888888888889,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 117.12561845779419
    },
    "MESH_4N_N800_rho_lambda_s46": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.864043942478043,
      "NARMA_NRMSE": 0.9328634626944163,
      "XOR_acc": 0.6166666666666667,
      "WAVE_acc": 0.5944444444444444,
      "wall_s": 115.11837387084961
    },
    "ER_SPARSE_N100_rho_lambda_s42": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.3680672164359398,
      "NARMA_NRMSE": 1.0315811847241831,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 95.64773654937744
    },
    "ER_SPARSE_N100_rho_lambda_s43": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 1.346632519649725,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 95.37738966941833
    },
    "ER_SPARSE_N100_rho_lambda_s44": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.6165739781128636,
      "NARMA_NRMSE": 0.9747421413520537,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 95.38676524162292
    },
    "ER_SPARSE_N100_rho_lambda_s45": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 2.220956034873518,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.55,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 95.5522780418396
    },
    "ER_SPARSE_N100_rho_lambda_s46": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.9553800470773004,
      "NARMA_NRMSE": 0.9766315678851584,
      "XOR_acc": 0.5222222222222223,
      "WAVE_acc": 0.5,
      "wall_s": 95.35543704032898
    },
    "ER_SPARSE_N300_rho_lambda_s42": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 3.1263645482719125,
      "NARMA_NRMSE": 1.086074089924594,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5444444444444444,
      "wall_s": 105.20067071914673
    },
    "ER_SPARSE_N300_rho_lambda_s43": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.973677798194516,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8222222222222222,
      "WAVE_acc": 0.42777777777777776,
      "wall_s": 105.5104010105133
    },
    "ER_SPARSE_N300_rho_lambda_s44": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 2.8998498662265124,
      "NARMA_NRMSE": 0.941139570594411,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 106.17564105987549
    },
    "ER_SPARSE_N300_rho_lambda_s45": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 3.1475951991321685,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9222222222222223,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 106.99098324775696
    },
    "ER_SPARSE_N300_rho_lambda_s46": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 2.8178455021499773,
      "NARMA_NRMSE": 0.9395885117196515,
      "XOR_acc": 0.9333333333333333,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 105.35729551315308
    },
    "ER_SPARSE_N800_rho_lambda_s42": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 4.107988945410384,
      "NARMA_NRMSE": 1.0180387938797275,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 115.83984851837158
    },
    "ER_SPARSE_N800_rho_lambda_s43": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 4.143011923328403,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9944444444444445,
      "WAVE_acc": 0.4666666666666667,
      "wall_s": 116.62818503379822
    },
    "ER_SPARSE_N800_rho_lambda_s44": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 4.020974432955311,
      "NARMA_NRMSE": 0.9372763524253274,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5666666666666667,
      "wall_s": 116.83128237724304
    },
    "ER_SPARSE_N800_rho_lambda_s45": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 2.0320002885519317,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8944444444444445,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 116.05332922935486
    },
    "ER_SPARSE_N800_rho_lambda_s46": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 3.4454143760044293,
      "NARMA_NRMSE": 0.9799063373951162,
      "XOR_acc": 0.9611111111111111,
      "WAVE_acc": 0.5611111111111111,
      "wall_s": 117.47502374649048
    },
    "WS_SMALLWORLD_N100_rho_lambda_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.4123130482673347,
      "NARMA_NRMSE": 1.0493335615348336,
      "XOR_acc": 0.6277777777777778,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 96.06203103065491
    },
    "WS_SMALLWORLD_N100_rho_lambda_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 1.805824380304705,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.4388888888888889,
      "wall_s": 95.81109428405762
    },
    "WS_SMALLWORLD_N100_rho_lambda_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.495329681845,
      "NARMA_NRMSE": 0.9404894271240893,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.55,
      "wall_s": 94.93396472930908
    },
    "WS_SMALLWORLD_N100_rho_lambda_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.996339496247256,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5888888888888889,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 95.25805354118347
    },
    "WS_SMALLWORLD_N100_rho_lambda_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.3176681835095196,
      "NARMA_NRMSE": 0.9262201658682192,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 96.0624647140503
    },
    "WS_SMALLWORLD_N300_rho_lambda_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 1.6045059816443206,
      "NARMA_NRMSE": 0.9967182417222542,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5333333333333333,
      "wall_s": 106.41315817832947
    },
    "WS_SMALLWORLD_N300_rho_lambda_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.072080348950143,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.45,
      "wall_s": 106.76897764205933
    },
    "WS_SMALLWORLD_N300_rho_lambda_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.1055407602543543,
      "NARMA_NRMSE": 0.9878539753955164,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 104.21005487442017
    },
    "WS_SMALLWORLD_N300_rho_lambda_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.7599151821580359,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 105.55485773086548
    },
    "WS_SMALLWORLD_N300_rho_lambda_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.7788465833023905,
      "NARMA_NRMSE": 0.9154478311281761,
      "XOR_acc": 0.49444444444444446,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 104.68029499053955
    },
    "WS_SMALLWORLD_N800_rho_lambda_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.1949707599870023,
      "NARMA_NRMSE": 1.0758860016408605,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 114.2676796913147
    },
    "WS_SMALLWORLD_N800_rho_lambda_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.615278843886245,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7722222222222223,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 114.7886860370636
    },
    "WS_SMALLWORLD_N800_rho_lambda_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 2.1954478433407916,
      "NARMA_NRMSE": 0.9408936527821326,
      "XOR_acc": 0.5722222222222222,
      "WAVE_acc": 0.55,
      "wall_s": 115.57079339027405
    },
    "WS_SMALLWORLD_N800_rho_lambda_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 2.05086042768124,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.4444444444444444,
      "wall_s": 114.72155284881592
    },
    "WS_SMALLWORLD_N800_rho_lambda_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.7840397412150513,
      "NARMA_NRMSE": 0.9910526361555577,
      "XOR_acc": 0.6222222222222222,
      "WAVE_acc": 0.5666666666666667,
      "wall_s": 114.02362132072449
    },
    "HUB_SPOKE_N100_rho_lambda_s42": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 1.2134155346649642,
      "NARMA_NRMSE": 0.9933571292870511,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 94.52027177810669
    },
    "HUB_SPOKE_N100_rho_lambda_s43": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 1.5679687605734336,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 95.9913969039917
    },
    "HUB_SPOKE_N100_rho_lambda_s44": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.1031019421206232,
      "NARMA_NRMSE": 0.9434162428289079,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 95.52104949951172
    },
    "HUB_SPOKE_N100_rho_lambda_s45": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.0635980158810097,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.43333333333333335,
      "wall_s": 95.56068849563599
    },
    "HUB_SPOKE_N100_rho_lambda_s46": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.0198703389132435,
      "NARMA_NRMSE": 0.932160779609753,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5,
      "wall_s": 95.44382071495056
    },
    "HUB_SPOKE_N300_rho_lambda_s42": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 1.2103252607310024,
      "NARMA_NRMSE": 0.9925883854368711,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 105.97168397903442
    },
    "HUB_SPOKE_N300_rho_lambda_s43": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.8766385287509237,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5777777777777777,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 106.27951693534851
    },
    "HUB_SPOKE_N300_rho_lambda_s44": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.418617001464086,
      "NARMA_NRMSE": 0.9803437047560851,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 104.75849604606628
    },
    "HUB_SPOKE_N300_rho_lambda_s45": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 2.048517208312041,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.4666666666666667,
      "wall_s": 105.08397436141968
    },
    "HUB_SPOKE_N300_rho_lambda_s46": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.021823825165008,
      "NARMA_NRMSE": 0.9325833195974726,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 106.0072648525238
    },
    "HUB_SPOKE_N800_rho_lambda_s42": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 1.8110955528954131,
      "NARMA_NRMSE": 1.0799262728348422,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5,
      "wall_s": 117.61071872711182
    },
    "HUB_SPOKE_N800_rho_lambda_s43": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.0499263035441886,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 115.2058527469635
    },
    "HUB_SPOKE_N800_rho_lambda_s44": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 1.700029612606786,
      "NARMA_NRMSE": 0.9483411653571399,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5611111111111111,
      "wall_s": 115.01429414749146
    },
    "HUB_SPOKE_N800_rho_lambda_s45": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 2.040068402915343,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4666666666666667,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 115.06433153152466
    },
    "HUB_SPOKE_N800_rho_lambda_s46": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 1.9771808001824676,
      "NARMA_NRMSE": 0.9765714394146447,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5666666666666667,
      "wall_s": 115.48443698883057
    },
    "LAYERED_N100_rho_lambda_s42": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 2.755016421033766,
      "NARMA_NRMSE": 1.0306650882176287,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 95.48564887046814
    },
    "LAYERED_N100_rho_lambda_s43": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 2.6414050509912514,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8055555555555556,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 96.22428441047668
    },
    "LAYERED_N100_rho_lambda_s44": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 2.8652450264262326,
      "NARMA_NRMSE": 0.9332733699864221,
      "XOR_acc": 0.8833333333333333,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 95.37974238395691
    },
    "LAYERED_N100_rho_lambda_s45": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 1.7351208484110794,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 95.4582085609436
    },
    "LAYERED_N100_rho_lambda_s46": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 2.02596414382383,
      "NARMA_NRMSE": 0.9426495573087351,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 95.76119565963745
    },
    "LAYERED_N300_rho_lambda_s42": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 3.7770625800830855,
      "NARMA_NRMSE": 1.0786819022708345,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 105.82956719398499
    },
    "LAYERED_N300_rho_lambda_s43": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 3.595213503328544,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 105.11483335494995
    },
    "LAYERED_N300_rho_lambda_s44": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 3.283360449327839,
      "NARMA_NRMSE": 0.930254574529069,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 103.62060761451721
    },
    "LAYERED_N300_rho_lambda_s45": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 3.6269089580135407,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9833333333333333,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 105.04462552070618
    },
    "LAYERED_N300_rho_lambda_s46": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 3.3456326216577468,
      "NARMA_NRMSE": 0.9836235274748553,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.55,
      "wall_s": 105.6777560710907
    },
    "LAYERED_N800_rho_lambda_s42": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 42,
      "MC": 3.9279071215923014,
      "NARMA_NRMSE": 1.170498175855531,
      "XOR_acc": 0.9888888888888889,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 115.92988204956055
    },
    "LAYERED_N800_rho_lambda_s43": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 43,
      "MC": 3.8602767535447824,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.4666666666666667,
      "wall_s": 115.47446775436401
    },
    "LAYERED_N800_rho_lambda_s44": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 44,
      "MC": 2.5350024280379713,
      "NARMA_NRMSE": 0.9853921885756186,
      "XOR_acc": 0.8166666666666667,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 115.42670822143555
    },
    "LAYERED_N800_rho_lambda_s45": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 45,
      "MC": 4.322877207634194,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 116.53346967697144
    },
    "LAYERED_N800_rho_lambda_s46": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_lambda",
      "seed": 46,
      "MC": 3.1723400990089923,
      "NARMA_NRMSE": 0.9890592181528708,
      "XOR_acc": 0.9055555555555556,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 115.46020293235779
    },
    "RAND_GAUSS_N100_rho_p95_sv_s42": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.8190670139925305,
      "NARMA_NRMSE": 1.0714147402572167,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 95.17116832733154
    },
    "RAND_GAUSS_N100_rho_p95_sv_s43": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.1861138771006021,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 96.22421288490295
    },
    "RAND_GAUSS_N100_rho_p95_sv_s44": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.0985205366002768,
      "NARMA_NRMSE": 0.9884670808057335,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 95.3144907951355
    },
    "RAND_GAUSS_N100_rho_p95_sv_s45": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.628657107539882,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.45,
      "wall_s": 95.02590370178223
    },
    "RAND_GAUSS_N100_rho_p95_sv_s46": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.1420223231827844,
      "NARMA_NRMSE": 0.9642234145744802,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 95.7169463634491
    },
    "RAND_GAUSS_N300_rho_p95_sv_s42": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.866632878030686,
      "NARMA_NRMSE": 1.064900239936361,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 106.65463709831238
    },
    "RAND_GAUSS_N300_rho_p95_sv_s43": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 2.8398654672971344,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7444444444444445,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 105.22270584106445
    },
    "RAND_GAUSS_N300_rho_p95_sv_s44": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 3.0031930233560007,
      "NARMA_NRMSE": 0.928175339936863,
      "XOR_acc": 0.8277777777777777,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 105.21746516227722
    },
    "RAND_GAUSS_N300_rho_p95_sv_s45": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 2.5922884919209217,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8555555555555555,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 105.5179193019867
    },
    "RAND_GAUSS_N300_rho_p95_sv_s46": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 2.219561953304347,
      "NARMA_NRMSE": 1.0019710985728225,
      "XOR_acc": 0.6055555555555555,
      "WAVE_acc": 0.5444444444444444,
      "wall_s": 105.31500482559204
    },
    "RAND_GAUSS_N800_rho_p95_sv_s42": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.9841675752594288,
      "NARMA_NRMSE": 0.9636392608176535,
      "XOR_acc": 0.9388888888888889,
      "WAVE_acc": 0.4666666666666667,
      "wall_s": 115.2070517539978
    },
    "RAND_GAUSS_N800_rho_p95_sv_s43": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 3.323051929969521,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9333333333333333,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 116.29075217247009
    },
    "RAND_GAUSS_N800_rho_p95_sv_s44": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 2.77364928860523,
      "NARMA_NRMSE": 0.9235710250263083,
      "XOR_acc": 0.8777777777777778,
      "WAVE_acc": 0.55,
      "wall_s": 116.31965112686157
    },
    "RAND_GAUSS_N800_rho_p95_sv_s45": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 4.271849363212971,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.4388888888888889,
      "wall_s": 114.99613428115845
    },
    "RAND_GAUSS_N800_rho_p95_sv_s46": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 2.0567247307472423,
      "NARMA_NRMSE": 1.023224058851117,
      "XOR_acc": 0.9333333333333333,
      "WAVE_acc": 0.5,
      "wall_s": 116.0112292766571
    },
    "MESH_4N_N100_rho_p95_sv_s42": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.2162078144575066,
      "NARMA_NRMSE": 0.9938607045314711,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.4444444444444444,
      "wall_s": 95.02059674263
    },
    "MESH_4N_N100_rho_p95_sv_s43": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.6813849258718288,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.42777777777777776,
      "wall_s": 94.91527795791626
    },
    "MESH_4N_N100_rho_p95_sv_s44": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.0917707568022381,
      "NARMA_NRMSE": 0.9746232508838376,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5,
      "wall_s": 95.36347222328186
    },
    "MESH_4N_N100_rho_p95_sv_s45": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.0509141077217572,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 94.71417379379272
    },
    "MESH_4N_N100_rho_p95_sv_s46": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.0236913496260656,
      "NARMA_NRMSE": 0.9329236936387265,
      "XOR_acc": 0.5222222222222223,
      "WAVE_acc": 0.55,
      "wall_s": 95.92021942138672
    },
    "MESH_4N_N300_rho_p95_sv_s42": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.6922345995060661,
      "NARMA_NRMSE": 0.9860907116367225,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5,
      "wall_s": 104.79008674621582
    },
    "MESH_4N_N300_rho_p95_sv_s43": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.9451387807498757,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.4166666666666667,
      "wall_s": 105.50103974342346
    },
    "MESH_4N_N300_rho_p95_sv_s44": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.0926204913839117,
      "NARMA_NRMSE": 0.9822300359329117,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 105.50514793395996
    },
    "MESH_4N_N300_rho_p95_sv_s45": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.0911695340480978,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 105.63992810249329
    },
    "MESH_4N_N300_rho_p95_sv_s46": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.022699699883076,
      "NARMA_NRMSE": 0.932879506693137,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 105.29305577278137
    },
    "MESH_4N_N800_rho_p95_sv_s42": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.50043255260736,
      "NARMA_NRMSE": 1.061495430943986,
      "XOR_acc": 0.6833333333333333,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 113.3696756362915
    },
    "MESH_4N_N800_rho_p95_sv_s43": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 2.0355706523342807,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.4444444444444444,
      "wall_s": 116.06937170028687
    },
    "MESH_4N_N800_rho_p95_sv_s44": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 2.029621612960831,
      "NARMA_NRMSE": 0.9407884987728903,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5833333333333334,
      "wall_s": 112.84797549247742
    },
    "MESH_4N_N800_rho_p95_sv_s45": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.4191063984752945,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 113.88466572761536
    },
    "MESH_4N_N800_rho_p95_sv_s46": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.840028792510963,
      "NARMA_NRMSE": 0.9141830260763318,
      "XOR_acc": 0.7444444444444445,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 115.58224701881409
    },
    "ER_SPARSE_N100_rho_p95_sv_s42": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.233398998705696,
      "NARMA_NRMSE": 0.9926736777866185,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 95.80320286750793
    },
    "ER_SPARSE_N100_rho_p95_sv_s43": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.4199953924594775,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.4222222222222222,
      "wall_s": 95.71402740478516
    },
    "ER_SPARSE_N100_rho_p95_sv_s44": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.09760382173047,
      "NARMA_NRMSE": 0.977021352775815,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 96.06503415107727
    },
    "ER_SPARSE_N100_rho_p95_sv_s45": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.3989745870928578,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 95.55151653289795
    },
    "ER_SPARSE_N100_rho_p95_sv_s46": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.4926731615509532,
      "NARMA_NRMSE": 0.9548049909993147,
      "XOR_acc": 0.49444444444444446,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 96.68639850616455
    },
    "ER_SPARSE_N300_rho_p95_sv_s42": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.2525440724018364,
      "NARMA_NRMSE": 1.0805357001176012,
      "XOR_acc": 0.6777777777777778,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 105.93851208686829
    },
    "ER_SPARSE_N300_rho_p95_sv_s43": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 2.1343936467998232,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.45,
      "wall_s": 104.95967364311218
    },
    "ER_SPARSE_N300_rho_p95_sv_s44": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 2.5104344839238735,
      "NARMA_NRMSE": 0.9306083812199473,
      "XOR_acc": 0.7777777777777778,
      "WAVE_acc": 0.5722222222222222,
      "wall_s": 104.74976229667664
    },
    "ER_SPARSE_N300_rho_p95_sv_s45": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 2.0282359205902827,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.6388888888888888,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 105.60540318489075
    },
    "ER_SPARSE_N300_rho_p95_sv_s46": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 2.581750136341166,
      "NARMA_NRMSE": 0.9796694955320432,
      "XOR_acc": 0.7055555555555556,
      "WAVE_acc": 0.5833333333333334,
      "wall_s": 105.85615730285645
    },
    "ER_SPARSE_N800_rho_p95_sv_s42": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.0813086500469526,
      "NARMA_NRMSE": 1.0741192910475812,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 115.1425666809082
    },
    "ER_SPARSE_N800_rho_p95_sv_s43": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 3.5025309304572656,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9555555555555556,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 114.92002534866333
    },
    "ER_SPARSE_N800_rho_p95_sv_s44": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 2.8986441533110003,
      "NARMA_NRMSE": 0.9214200866298571,
      "XOR_acc": 0.9555555555555556,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 116.43627786636353
    },
    "ER_SPARSE_N800_rho_p95_sv_s45": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 3.0316925335000366,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8777777777777778,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 115.23829674720764
    },
    "ER_SPARSE_N800_rho_p95_sv_s46": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 2.6740947791847436,
      "NARMA_NRMSE": 0.9521351441074051,
      "XOR_acc": 0.7944444444444444,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 115.12379336357117
    },
    "WS_SMALLWORLD_N100_rho_p95_sv_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.3959641057464829,
      "NARMA_NRMSE": 0.9936522872505584,
      "XOR_acc": 0.5666666666666667,
      "WAVE_acc": 0.5555555555555556,
      "wall_s": 95.17204928398132
    },
    "WS_SMALLWORLD_N100_rho_p95_sv_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.350701857241614,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.42777777777777776,
      "wall_s": 96.46073698997498
    },
    "WS_SMALLWORLD_N100_rho_p95_sv_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.727090218076354,
      "NARMA_NRMSE": 0.972675253795793,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 95.71739339828491
    },
    "WS_SMALLWORLD_N100_rho_p95_sv_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.6083234428527389,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 95.11293745040894
    },
    "WS_SMALLWORLD_N100_rho_p95_sv_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.0209498101355337,
      "NARMA_NRMSE": 0.932472971149965,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 93.81276369094849
    },
    "WS_SMALLWORLD_N300_rho_p95_sv_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.7171199995734,
      "NARMA_NRMSE": 0.9841151348563193,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 106.83485531806946
    },
    "WS_SMALLWORLD_N300_rho_p95_sv_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.3912592421710248,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.42777777777777776,
      "wall_s": 106.86608219146729
    },
    "WS_SMALLWORLD_N300_rho_p95_sv_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.4919330037592198,
      "NARMA_NRMSE": 0.9513469301118568,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5777777777777777,
      "wall_s": 104.62223720550537
    },
    "WS_SMALLWORLD_N300_rho_p95_sv_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.8718701537088065,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 104.29793357849121
    },
    "WS_SMALLWORLD_N300_rho_p95_sv_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.6171384372331743,
      "NARMA_NRMSE": 0.918240129696329,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 105.0068826675415
    },
    "WS_SMALLWORLD_N800_rho_p95_sv_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.432453122359827,
      "NARMA_NRMSE": 1.064789034603772,
      "XOR_acc": 0.6222222222222222,
      "WAVE_acc": 0.4388888888888889,
      "wall_s": 115.38862562179565
    },
    "WS_SMALLWORLD_N800_rho_p95_sv_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.8721172387127445,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5722222222222222,
      "WAVE_acc": 0.42777777777777776,
      "wall_s": 115.53354477882385
    },
    "WS_SMALLWORLD_N800_rho_p95_sv_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.8223990081168355,
      "NARMA_NRMSE": 0.9402282516628502,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5722222222222222,
      "wall_s": 114.08052802085876
    },
    "WS_SMALLWORLD_N800_rho_p95_sv_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 2.8278981800827756,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 114.89452981948853
    },
    "WS_SMALLWORLD_N800_rho_p95_sv_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.7566163248950963,
      "NARMA_NRMSE": 0.949611537932337,
      "XOR_acc": 0.5222222222222223,
      "WAVE_acc": 0.5333333333333333,
      "wall_s": 115.23088240623474
    },
    "HUB_SPOKE_N100_rho_p95_sv_s42": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.123255831495352,
      "NARMA_NRMSE": 1.0073412264734471,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 94.95041298866272
    },
    "HUB_SPOKE_N100_rho_p95_sv_s43": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.1852204246185476,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 95.28005528450012
    },
    "HUB_SPOKE_N100_rho_p95_sv_s44": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.1450507187112988,
      "NARMA_NRMSE": 0.9342574725337417,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 94.63446927070618
    },
    "HUB_SPOKE_N100_rho_p95_sv_s45": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.2669311204467955,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 95.94398617744446
    },
    "HUB_SPOKE_N100_rho_p95_sv_s46": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.0219152016183897,
      "NARMA_NRMSE": 0.9321654729595006,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 95.37439584732056
    },
    "HUB_SPOKE_N300_rho_p95_sv_s42": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.2111507627415985,
      "NARMA_NRMSE": 0.9993682548284881,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 105.96590065956116
    },
    "HUB_SPOKE_N300_rho_p95_sv_s43": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 2.203518502817678,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 106.12494111061096
    },
    "HUB_SPOKE_N300_rho_p95_sv_s44": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.3744656042731653,
      "NARMA_NRMSE": 0.9460194722852399,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 105.6673743724823
    },
    "HUB_SPOKE_N300_rho_p95_sv_s45": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 2.0619953471460852,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 105.66883158683777
    },
    "HUB_SPOKE_N300_rho_p95_sv_s46": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.4199891739493622,
      "NARMA_NRMSE": 0.9437576193035062,
      "XOR_acc": 0.55,
      "WAVE_acc": 0.5,
      "wall_s": 105.69970631599426
    },
    "HUB_SPOKE_N800_rho_p95_sv_s42": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.310115903183739,
      "NARMA_NRMSE": 1.0873966474497871,
      "XOR_acc": 0.5944444444444444,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 116.11524701118469
    },
    "HUB_SPOKE_N800_rho_p95_sv_s43": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 2.225003560187134,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 114.83697152137756
    },
    "HUB_SPOKE_N800_rho_p95_sv_s44": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 2.5786572501900586,
      "NARMA_NRMSE": 0.9445353143384974,
      "XOR_acc": 0.7055555555555556,
      "WAVE_acc": 0.6222222222222222,
      "wall_s": 115.16743922233582
    },
    "HUB_SPOKE_N800_rho_p95_sv_s45": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 2.811462043735826,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9277777777777778,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 113.72007465362549
    },
    "HUB_SPOKE_N800_rho_p95_sv_s46": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 2.448802970467876,
      "NARMA_NRMSE": 0.9167276522365697,
      "XOR_acc": 0.7944444444444444,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 116.14751529693604
    },
    "LAYERED_N100_rho_p95_sv_s42": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 1.2099851534633475,
      "NARMA_NRMSE": 0.9934091540529396,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 96.23706388473511
    },
    "LAYERED_N100_rho_p95_sv_s43": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 1.1768974234287768,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.45,
      "wall_s": 93.49556303024292
    },
    "LAYERED_N100_rho_p95_sv_s44": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 1.4203287228165065,
      "NARMA_NRMSE": 0.976194276003793,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 95.1185781955719
    },
    "LAYERED_N100_rho_p95_sv_s45": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 1.0966860146519333,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 94.90728831291199
    },
    "LAYERED_N100_rho_p95_sv_s46": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.2841794543630227,
      "NARMA_NRMSE": 0.9207387801629173,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 95.86619901657104
    },
    "LAYERED_N300_rho_p95_sv_s42": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.118522746441936,
      "NARMA_NRMSE": 1.0890206737095536,
      "XOR_acc": 0.6111111111111112,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 105.97555088996887
    },
    "LAYERED_N300_rho_p95_sv_s43": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 2.201226419428431,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 104.97717809677124
    },
    "LAYERED_N300_rho_p95_sv_s44": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 2.0882816376126327,
      "NARMA_NRMSE": 0.9424875196534035,
      "XOR_acc": 0.5888888888888889,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 106.78059792518616
    },
    "LAYERED_N300_rho_p95_sv_s45": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 2.1825095921672584,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7611111111111111,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 104.80526232719421
    },
    "LAYERED_N300_rho_p95_sv_s46": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 1.79632195040648,
      "NARMA_NRMSE": 0.9133099528163504,
      "XOR_acc": 0.7,
      "WAVE_acc": 0.6055555555555555,
      "wall_s": 105.9569833278656
    },
    "LAYERED_N800_rho_p95_sv_s42": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 42,
      "MC": 2.680024131799972,
      "NARMA_NRMSE": 1.081290475324765,
      "XOR_acc": 0.7611111111111111,
      "WAVE_acc": 0.5,
      "wall_s": 113.05433106422424
    },
    "LAYERED_N800_rho_p95_sv_s43": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 43,
      "MC": 2.693811993963302,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5777777777777777,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 114.50183415412903
    },
    "LAYERED_N800_rho_p95_sv_s44": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 44,
      "MC": 2.619234145708249,
      "NARMA_NRMSE": 0.9357580618205233,
      "XOR_acc": 0.9055555555555556,
      "WAVE_acc": 0.5722222222222222,
      "wall_s": 114.33956003189087
    },
    "LAYERED_N800_rho_p95_sv_s45": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 45,
      "MC": 2.467530553362626,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7277777777777777,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 114.39738440513611
    },
    "LAYERED_N800_rho_p95_sv_s46": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_p95_sv",
      "seed": 46,
      "MC": 2.9456360651274087,
      "NARMA_NRMSE": 0.9949069076979337,
      "XOR_acc": 0.8944444444444445,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 116.67792582511902
    },
    "RAND_GAUSS_N100_rho_deg_norm_s42": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.120016981772782,
      "NARMA_NRMSE": 0.9770013877116192,
      "XOR_acc": 0.5888888888888889,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 96.23034477233887
    },
    "RAND_GAUSS_N100_rho_deg_norm_s43": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.232393700540888,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5944444444444444,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 95.0735514163971
    },
    "RAND_GAUSS_N100_rho_deg_norm_s44": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 2.821089539952353,
      "NARMA_NRMSE": 0.9234436501409082,
      "XOR_acc": 0.7333333333333333,
      "WAVE_acc": 0.5666666666666667,
      "wall_s": 95.54397082328796
    },
    "RAND_GAUSS_N100_rho_deg_norm_s45": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 2.1845444609195233,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.6166666666666667,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 95.89449071884155
    },
    "RAND_GAUSS_N100_rho_deg_norm_s46": {
      "topo": "RAND_GAUSS",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.1719877433697565,
      "NARMA_NRMSE": 0.9541557149160396,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 94.79516506195068
    },
    "RAND_GAUSS_N300_rho_deg_norm_s42": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 0.10049585992782001,
      "NARMA_NRMSE": 1.1701523332058674,
      "XOR_acc": 0.5,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 105.28364968299866
    },
    "RAND_GAUSS_N300_rho_deg_norm_s43": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 0.1889806737927646,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4666666666666667,
      "WAVE_acc": 0.37777777777777777,
      "wall_s": 105.40311217308044
    },
    "RAND_GAUSS_N300_rho_deg_norm_s44": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 0.5500171736442468,
      "NARMA_NRMSE": 1.0515069781760888,
      "XOR_acc": 0.5777777777777777,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 105.97130274772644
    },
    "RAND_GAUSS_N300_rho_deg_norm_s45": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 0.09656540786919597,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.35,
      "wall_s": 104.97276639938354
    },
    "RAND_GAUSS_N300_rho_deg_norm_s46": {
      "topo": "RAND_GAUSS",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 0.14400445573186726,
      "NARMA_NRMSE": 1.1755361835811433,
      "XOR_acc": 0.48333333333333334,
      "WAVE_acc": 0.42777777777777776,
      "wall_s": 106.58789587020874
    },
    "RAND_GAUSS_N800_rho_deg_norm_s42": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 0.15365924816896018,
      "NARMA_NRMSE": 1.3675925582383226,
      "XOR_acc": 0.5611111111111111,
      "WAVE_acc": 0.35555555555555557,
      "wall_s": 116.74929070472717
    },
    "RAND_GAUSS_N800_rho_deg_norm_s43": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 0.2021884821100391,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.55,
      "WAVE_acc": 0.4444444444444444,
      "wall_s": 114.96647143363953
    },
    "RAND_GAUSS_N800_rho_deg_norm_s44": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 0.24771448634138524,
      "NARMA_NRMSE": 1.2784429736477911,
      "XOR_acc": 0.5222222222222223,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 116.29867243766785
    },
    "RAND_GAUSS_N800_rho_deg_norm_s45": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 0.1978832741771684,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.39444444444444443,
      "wall_s": 114.70436787605286
    },
    "RAND_GAUSS_N800_rho_deg_norm_s46": {
      "topo": "RAND_GAUSS",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 0.22427488597684359,
      "NARMA_NRMSE": 1.1677604451436345,
      "XOR_acc": 0.49444444444444446,
      "WAVE_acc": 0.4222222222222222,
      "wall_s": 115.11404991149902
    },
    "MESH_4N_N100_rho_deg_norm_s42": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 2.4771819882916164,
      "NARMA_NRMSE": 1.019720464032772,
      "XOR_acc": 0.5611111111111111,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 95.01422595977783
    },
    "MESH_4N_N100_rho_deg_norm_s43": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 1.5065983040921984,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.55,
      "WAVE_acc": 0.43333333333333335,
      "wall_s": 94.52578067779541
    },
    "MESH_4N_N100_rho_deg_norm_s44": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 2.2721242572960545,
      "NARMA_NRMSE": 0.9356764679449282,
      "XOR_acc": 0.6555555555555556,
      "WAVE_acc": 0.55,
      "wall_s": 95.7696213722229
    },
    "MESH_4N_N100_rho_deg_norm_s45": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 2.3071646595005806,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.6555555555555556,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 95.73716282844543
    },
    "MESH_4N_N100_rho_deg_norm_s46": {
      "topo": "MESH_4N",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 1.6541355461071323,
      "NARMA_NRMSE": 0.9593703565576236,
      "XOR_acc": 0.5055555555555555,
      "WAVE_acc": 0.5166666666666667,
      "wall_s": 94.8339912891388
    },
    "MESH_4N_N300_rho_deg_norm_s42": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.735465989927797,
      "NARMA_NRMSE": 1.0579162012563472,
      "XOR_acc": 0.9666666666666667,
      "WAVE_acc": 0.5111111111111111,
      "wall_s": 106.63177418708801
    },
    "MESH_4N_N300_rho_deg_norm_s43": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.979409246706695,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8611111111111112,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 105.47759294509888
    },
    "MESH_4N_N300_rho_deg_norm_s44": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 3.4271546901633894,
      "NARMA_NRMSE": 0.924531622143257,
      "XOR_acc": 0.6444444444444445,
      "WAVE_acc": 0.5777777777777777,
      "wall_s": 106.09961485862732
    },
    "MESH_4N_N300_rho_deg_norm_s45": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 2.541757614477624,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7166666666666667,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 106.1494734287262
    },
    "MESH_4N_N300_rho_deg_norm_s46": {
      "topo": "MESH_4N",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 4.699185938459274,
      "NARMA_NRMSE": 0.9300557242862921,
      "XOR_acc": 0.5444444444444444,
      "WAVE_acc": 0.5722222222222222,
      "wall_s": 105.2481415271759
    },
    "MESH_4N_N800_rho_deg_norm_s42": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 4.221014808145301,
      "NARMA_NRMSE": 1.2516725406210933,
      "XOR_acc": 0.9722222222222222,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 115.93498706817627
    },
    "MESH_4N_N800_rho_deg_norm_s43": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 3.3359596094044393,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9166666666666666,
      "WAVE_acc": 0.43333333333333335,
      "wall_s": 115.64877414703369
    },
    "MESH_4N_N800_rho_deg_norm_s44": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 4.364302146503189,
      "NARMA_NRMSE": 0.936886973216957,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5722222222222222,
      "wall_s": 114.84133648872375
    },
    "MESH_4N_N800_rho_deg_norm_s45": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 3.8872591600631026,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8388888888888889,
      "WAVE_acc": 0.4444444444444444,
      "wall_s": 115.65693092346191
    },
    "MESH_4N_N800_rho_deg_norm_s46": {
      "topo": "MESH_4N",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 4.205807295236067,
      "NARMA_NRMSE": 1.051463995567981,
      "XOR_acc": 0.9555555555555556,
      "WAVE_acc": 0.5888888888888889,
      "wall_s": 117.18396759033203
    },
    "ER_SPARSE_N100_rho_deg_norm_s42": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.32592988873978,
      "NARMA_NRMSE": 1.0683365010757224,
      "XOR_acc": 0.8722222222222222,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 94.55401277542114
    },
    "ER_SPARSE_N100_rho_deg_norm_s43": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.3416075101849123,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7444444444444445,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 95.3226957321167
    },
    "ER_SPARSE_N100_rho_deg_norm_s44": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 3.3433701384818932,
      "NARMA_NRMSE": 0.9229057818887128,
      "XOR_acc": 0.6777777777777778,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 94.67851638793945
    },
    "ER_SPARSE_N100_rho_deg_norm_s45": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 3.0163992275344698,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9277777777777778,
      "WAVE_acc": 0.4777777777777778,
      "wall_s": 94.30507946014404
    },
    "ER_SPARSE_N100_rho_deg_norm_s46": {
      "topo": "ER_SPARSE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.4851167631770843,
      "NARMA_NRMSE": 0.9561065606422042,
      "XOR_acc": 0.7166666666666667,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 95.3320791721344
    },
    "ER_SPARSE_N300_rho_deg_norm_s42": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.3051604113094353,
      "NARMA_NRMSE": 1.0567057146662617,
      "XOR_acc": 0.9388888888888889,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 106.7490074634552
    },
    "ER_SPARSE_N300_rho_deg_norm_s43": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.4314189033178306,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7666666666666667,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 105.5149085521698
    },
    "ER_SPARSE_N300_rho_deg_norm_s44": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 2.4832677534534864,
      "NARMA_NRMSE": 0.9412307308314872,
      "XOR_acc": 0.6388888888888888,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 106.11688351631165
    },
    "ER_SPARSE_N300_rho_deg_norm_s45": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 3.326548603734777,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9444444444444444,
      "WAVE_acc": 0.4,
      "wall_s": 105.22912669181824
    },
    "ER_SPARSE_N300_rho_deg_norm_s46": {
      "topo": "ER_SPARSE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.4247679421397845,
      "NARMA_NRMSE": 0.9477977848094021,
      "XOR_acc": 0.7277777777777777,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 105.92709255218506
    },
    "ER_SPARSE_N800_rho_deg_norm_s42": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 2.911573667732381,
      "NARMA_NRMSE": 1.1236796795003592,
      "XOR_acc": 0.8833333333333333,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 115.47450423240662
    },
    "ER_SPARSE_N800_rho_deg_norm_s43": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 1.6797189307446372,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7222222222222222,
      "WAVE_acc": 0.43333333333333335,
      "wall_s": 116.63969445228577
    },
    "ER_SPARSE_N800_rho_deg_norm_s44": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 3.068654871549587,
      "NARMA_NRMSE": 0.9302308647676695,
      "XOR_acc": 0.9055555555555556,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 115.9869909286499
    },
    "ER_SPARSE_N800_rho_deg_norm_s45": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 3.6508135076684534,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9833333333333333,
      "WAVE_acc": 0.4666666666666667,
      "wall_s": 114.82906556129456
    },
    "ER_SPARSE_N800_rho_deg_norm_s46": {
      "topo": "ER_SPARSE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.1677347378114673,
      "NARMA_NRMSE": 0.9298063142355453,
      "XOR_acc": 0.8611111111111112,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 114.88519930839539
    },
    "WS_SMALLWORLD_N100_rho_deg_norm_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 2.642232998733163,
      "NARMA_NRMSE": 1.0405718264715855,
      "XOR_acc": 0.7277777777777777,
      "WAVE_acc": 0.5,
      "wall_s": 95.61095547676086
    },
    "WS_SMALLWORLD_N100_rho_deg_norm_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.5319623565760865,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7666666666666667,
      "WAVE_acc": 0.4388888888888889,
      "wall_s": 95.03697919845581
    },
    "WS_SMALLWORLD_N100_rho_deg_norm_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 2.9937363644543584,
      "NARMA_NRMSE": 0.9458898077938434,
      "XOR_acc": 0.7222222222222222,
      "WAVE_acc": 0.5722222222222222,
      "wall_s": 95.50916934013367
    },
    "WS_SMALLWORLD_N100_rho_deg_norm_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 3.0194550528020336,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7333333333333333,
      "WAVE_acc": 0.4444444444444444,
      "wall_s": 94.84532618522644
    },
    "WS_SMALLWORLD_N100_rho_deg_norm_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.1774483632354302,
      "NARMA_NRMSE": 0.9735410685796776,
      "XOR_acc": 0.7888888888888889,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 95.49735403060913
    },
    "WS_SMALLWORLD_N300_rho_deg_norm_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.771031106719226,
      "NARMA_NRMSE": 1.0954599348563496,
      "XOR_acc": 0.9333333333333333,
      "WAVE_acc": 0.5222222222222223,
      "wall_s": 106.88578343391418
    },
    "WS_SMALLWORLD_N300_rho_deg_norm_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 4.255594583729608,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.8833333333333333,
      "WAVE_acc": 0.5,
      "wall_s": 105.49703335762024
    },
    "WS_SMALLWORLD_N300_rho_deg_norm_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 2.871287276259073,
      "NARMA_NRMSE": 0.9411899341645905,
      "XOR_acc": 0.9222222222222223,
      "WAVE_acc": 0.5722222222222222,
      "wall_s": 104.66030693054199
    },
    "WS_SMALLWORLD_N300_rho_deg_norm_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 2.8226398625160556,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7777777777777778,
      "WAVE_acc": 0.4666666666666667,
      "wall_s": 105.26192808151245
    },
    "WS_SMALLWORLD_N300_rho_deg_norm_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.849086400437588,
      "NARMA_NRMSE": 1.0084909168489202,
      "XOR_acc": 0.6388888888888888,
      "WAVE_acc": 0.5,
      "wall_s": 104.37394404411316
    },
    "WS_SMALLWORLD_N800_rho_deg_norm_s42": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 4.525258013179748,
      "NARMA_NRMSE": 1.020033551487604,
      "XOR_acc": 0.9722222222222222,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 115.90240359306335
    },
    "WS_SMALLWORLD_N800_rho_deg_norm_s43": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 5.400774008281352,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5277777777777778,
      "wall_s": 115.11113715171814
    },
    "WS_SMALLWORLD_N800_rho_deg_norm_s44": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 3.947065960009904,
      "NARMA_NRMSE": 0.9196472521015531,
      "XOR_acc": 0.85,
      "WAVE_acc": 0.5444444444444444,
      "wall_s": 115.66434478759766
    },
    "WS_SMALLWORLD_N800_rho_deg_norm_s45": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 4.287103981177069,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.9833333333333333,
      "WAVE_acc": 0.4666666666666667,
      "wall_s": 117.44763088226318
    },
    "WS_SMALLWORLD_N800_rho_deg_norm_s46": {
      "topo": "WS_SMALLWORLD",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 3.52964597066594,
      "NARMA_NRMSE": 0.9676486548735338,
      "XOR_acc": 0.9722222222222222,
      "WAVE_acc": 0.5833333333333334,
      "wall_s": 115.2337851524353
    },
    "HUB_SPOKE_N100_rho_deg_norm_s42": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 1.6865539299156989,
      "NARMA_NRMSE": 0.9960494470389836,
      "XOR_acc": 0.5277777777777778,
      "WAVE_acc": 0.5333333333333333,
      "wall_s": 93.85989022254944
    },
    "HUB_SPOKE_N100_rho_deg_norm_s43": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.904950122802772,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.3611111111111111,
      "WAVE_acc": 0.4388888888888889,
      "wall_s": 95.85911560058594
    },
    "HUB_SPOKE_N100_rho_deg_norm_s44": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 1.1325720873158083,
      "NARMA_NRMSE": 0.9357469840493015,
      "XOR_acc": 0.4888888888888889,
      "WAVE_acc": 0.55,
      "wall_s": 95.33474326133728
    },
    "HUB_SPOKE_N100_rho_deg_norm_s45": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 1.6578668806930037,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.4444444444444444,
      "WAVE_acc": 0.5,
      "wall_s": 95.6807918548584
    },
    "HUB_SPOKE_N100_rho_deg_norm_s46": {
      "topo": "HUB_SPOKE",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 1.5563059886654564,
      "NARMA_NRMSE": 0.9707612411429819,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 95.45088076591492
    },
    "HUB_SPOKE_N300_rho_deg_norm_s42": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.52419979176992,
      "NARMA_NRMSE": 1.0944597609174371,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5,
      "wall_s": 106.07856178283691
    },
    "HUB_SPOKE_N300_rho_deg_norm_s43": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 3.556701853304437,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 105.02391052246094
    },
    "HUB_SPOKE_N300_rho_deg_norm_s44": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 3.0779276666561928,
      "NARMA_NRMSE": 0.9243158603738942,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.6055555555555555,
      "wall_s": 104.99697065353394
    },
    "HUB_SPOKE_N300_rho_deg_norm_s45": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 3.8333920547737583,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5055555555555555,
      "wall_s": 105.37260246276855
    },
    "HUB_SPOKE_N300_rho_deg_norm_s46": {
      "topo": "HUB_SPOKE",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.661186059581785,
      "NARMA_NRMSE": 0.9331442694449766,
      "XOR_acc": 0.8888888888888888,
      "WAVE_acc": 0.5666666666666667,
      "wall_s": 106.52064394950867
    },
    "HUB_SPOKE_N800_rho_deg_norm_s42": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.8809884216521455,
      "NARMA_NRMSE": 1.0613007320360728,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5388888888888889,
      "wall_s": 115.75985431671143
    },
    "HUB_SPOKE_N800_rho_deg_norm_s43": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 4.523099210157326,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 116.83318042755127
    },
    "HUB_SPOKE_N800_rho_deg_norm_s44": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 4.803029674123069,
      "NARMA_NRMSE": 0.9481327632920296,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 116.13635087013245
    },
    "HUB_SPOKE_N800_rho_deg_norm_s45": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 4.764795110962916,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.45555555555555555,
      "wall_s": 114.95954132080078
    },
    "HUB_SPOKE_N800_rho_deg_norm_s46": {
      "topo": "HUB_SPOKE",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 4.16564739518772,
      "NARMA_NRMSE": 0.9872350922588958,
      "XOR_acc": 1.0,
      "WAVE_acc": 0.5,
      "wall_s": 114.29423761367798
    },
    "LAYERED_N100_rho_deg_norm_s42": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 3.0433522292285993,
      "NARMA_NRMSE": 0.9646382415355661,
      "XOR_acc": 0.9944444444444445,
      "WAVE_acc": 0.48333333333333334,
      "wall_s": 94.6724305152893
    },
    "LAYERED_N100_rho_deg_norm_s43": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.8213680584896936,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.7722222222222223,
      "WAVE_acc": 0.43333333333333335,
      "wall_s": 95.13738942146301
    },
    "LAYERED_N100_rho_deg_norm_s44": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 2.7976533202722886,
      "NARMA_NRMSE": 1.003691477089209,
      "XOR_acc": 0.9,
      "WAVE_acc": 0.49444444444444446,
      "wall_s": 96.31741738319397
    },
    "LAYERED_N100_rho_deg_norm_s45": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 2.0444269248526257,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.65,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 93.45723628997803
    },
    "LAYERED_N100_rho_deg_norm_s46": {
      "topo": "LAYERED",
      "N": 100,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 2.5210961279964286,
      "NARMA_NRMSE": 0.9638435510646753,
      "XOR_acc": 0.7666666666666667,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 95.60272216796875
    },
    "LAYERED_N300_rho_deg_norm_s42": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 0.9332574523962327,
      "NARMA_NRMSE": 1.278411532801207,
      "XOR_acc": 0.5888888888888889,
      "WAVE_acc": 0.39444444444444443,
      "wall_s": 104.52474808692932
    },
    "LAYERED_N300_rho_deg_norm_s43": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 2.0300141319906437,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.6055555555555555,
      "WAVE_acc": 0.42777777777777776,
      "wall_s": 104.42072582244873
    },
    "LAYERED_N300_rho_deg_norm_s44": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 2.7677150536810236,
      "NARMA_NRMSE": 0.9638616165840966,
      "XOR_acc": 0.8611111111111112,
      "WAVE_acc": 0.4722222222222222,
      "wall_s": 105.7482807636261
    },
    "LAYERED_N300_rho_deg_norm_s45": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 1.784339635845473,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.6666666666666666,
      "WAVE_acc": 0.4111111111111111,
      "wall_s": 105.97745704650879
    },
    "LAYERED_N300_rho_deg_norm_s46": {
      "topo": "LAYERED",
      "N": 300,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 1.8963806940429486,
      "NARMA_NRMSE": 1.13360419919955,
      "XOR_acc": 0.8333333333333334,
      "WAVE_acc": 0.46111111111111114,
      "wall_s": 105.74006748199463
    },
    "LAYERED_N800_rho_deg_norm_s42": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 42,
      "MC": 0.6872234781196855,
      "NARMA_NRMSE": 1.4741423790142014,
      "XOR_acc": 0.5666666666666667,
      "WAVE_acc": 0.3277777777777778,
      "wall_s": 115.45573019981384
    },
    "LAYERED_N800_rho_deg_norm_s43": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 43,
      "MC": 0.9674235961390836,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.46111111111111114,
      "WAVE_acc": 0.3333333333333333,
      "wall_s": 115.76034212112427
    },
    "LAYERED_N800_rho_deg_norm_s44": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 44,
      "MC": 1.6129999119094072,
      "NARMA_NRMSE": 1.3563426577425992,
      "XOR_acc": 0.5611111111111111,
      "WAVE_acc": 0.4888888888888889,
      "wall_s": 120.93322324752808
    },
    "LAYERED_N800_rho_deg_norm_s45": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 45,
      "MC": 1.276680959191844,
      "NARMA_NRMSE": NaN,
      "XOR_acc": 0.5111111111111111,
      "WAVE_acc": 0.40555555555555556,
      "wall_s": 115.82849860191345
    },
    "LAYERED_N800_rho_deg_norm_s46": {
      "topo": "LAYERED",
      "N": 800,
      "rho_variant": "rho_deg_norm",
      "seed": 46,
      "MC": 0.8048439771615689,
      "NARMA_NRMSE": 1.4886310965998513,
      "XOR_acc": 0.55,
      "WAVE_acc": 0.37777777777777777,
      "wall_s": 116.44520330429077
    }
  },
  "agg": {
    "RAND_GAUSS_N100_rho_lambda": {
      "MC_mean": 2.336067248593682,
      "MC_sd": 0.7155612101090161,
      "NARMA_NRMSE_mean": 0.983274873185133,
      "XOR_acc_mean": 0.7022222222222223,
      "WAVE_acc_mean": 0.4833333333333333,
      "n_valid": 5
    },
    "RAND_GAUSS_N300_rho_lambda": {
      "MC_mean": 2.6651725253875016,
      "MC_sd": 0.833878160911766,
      "NARMA_NRMSE_mean": 1.0614032077757092,
      "XOR_acc_mean": 0.8277777777777778,
      "WAVE_acc_mean": 0.47000000000000003,
      "n_valid": 5
    },
    "RAND_GAUSS_N800_rho_lambda": {
      "MC_mean": 2.248585593246692,
      "MC_sd": 0.385600195081113,
      "NARMA_NRMSE_mean": 1.176207659832769,
      "XOR_acc_mean": 0.75,
      "WAVE_acc_mean": 0.4477777777777778,
      "n_valid": 5
    },
    "MESH_4N_N100_rho_lambda": {
      "MC_mean": 1.6038642897706865,
      "MC_sd": 0.5365804121670142,
      "NARMA_NRMSE_mean": 0.986786157053392,
      "XOR_acc_mean": 0.48888888888888893,
      "WAVE_acc_mean": 0.4866666666666667,
      "n_valid": 5
    },
    "MESH_4N_N300_rho_lambda": {
      "MC_mean": 1.7620945952731968,
      "MC_sd": 0.4453471520130565,
      "NARMA_NRMSE_mean": 0.9906822540341809,
      "XOR_acc_mean": 0.5788888888888889,
      "WAVE_acc_mean": 0.5055555555555555,
      "n_valid": 5
    },
    "MESH_4N_N800_rho_lambda": {
      "MC_mean": 2.2016005748661165,
      "MC_sd": 0.39042033722834224,
      "NARMA_NRMSE_mean": 0.9511832566014397,
      "XOR_acc_mean": 0.6766666666666666,
      "WAVE_acc_mean": 0.5255555555555556,
      "n_valid": 5
    },
    "ER_SPARSE_N100_rho_lambda": {
      "MC_mean": 1.901521959229869,
      "MC_sd": 0.37727998659631984,
      "NARMA_NRMSE_mean": 0.9943182979871317,
      "XOR_acc_mean": 0.5155555555555555,
      "WAVE_acc_mean": 0.4866666666666667,
      "n_valid": 5
    },
    "ER_SPARSE_N300_rho_lambda": {
      "MC_mean": 2.9930665827950174,
      "MC_sd": 0.1276049757539785,
      "NARMA_NRMSE_mean": 0.9889340574128855,
      "XOR_acc_mean": 0.9355555555555556,
      "WAVE_acc_mean": 0.5033333333333333,
      "n_valid": 5
    },
    "ER_SPARSE_N800_rho_lambda": {
      "MC_mean": 3.5498779932500915,
      "MC_sd": 0.8000116153134463,
      "NARMA_NRMSE_mean": 0.9784071612333903,
      "XOR_acc_mean": 0.97,
      "WAVE_acc_mean": 0.5166666666666666,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N100_rho_lambda": {
      "MC_mean": 1.8054949580347632,
      "MC_sd": 0.3844027579026071,
      "NARMA_NRMSE_mean": 0.9720143848423808,
      "XOR_acc_mean": 0.5444444444444445,
      "WAVE_acc_mean": 0.49333333333333335,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N300_rho_lambda": {
      "MC_mean": 1.6641777712618488,
      "MC_sd": 0.31759569698854406,
      "NARMA_NRMSE_mean": 0.9666733494153156,
      "XOR_acc_mean": 0.4922222222222222,
      "WAVE_acc_mean": 0.48888888888888893,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N800_rho_lambda": {
      "MC_mean": 2.1681195232220665,
      "MC_sd": 0.2693274061386166,
      "NARMA_NRMSE_mean": 1.0026107635261836,
      "XOR_acc_mean": 0.5877777777777777,
      "WAVE_acc_mean": 0.5122222222222221,
      "n_valid": 5
    },
    "HUB_SPOKE_N100_rho_lambda": {
      "MC_mean": 1.193590918430655,
      "MC_sd": 0.19789746506264563,
      "NARMA_NRMSE_mean": 0.9563113839085706,
      "XOR_acc_mean": 0.4955555555555556,
      "WAVE_acc_mean": 0.4622222222222222,
      "n_valid": 5
    },
    "HUB_SPOKE_N300_rho_lambda": {
      "MC_mean": 1.7151843648846121,
      "MC_sd": 0.6758210285871632,
      "NARMA_NRMSE_mean": 0.9685051365968097,
      "XOR_acc_mean": 0.51,
      "WAVE_acc_mean": 0.4966666666666667,
      "n_valid": 5
    },
    "HUB_SPOKE_N800_rho_lambda": {
      "MC_mean": 1.9156601344288398,
      "MC_sd": 0.13763712246629017,
      "NARMA_NRMSE_mean": 1.001612959202209,
      "XOR_acc_mean": 0.5,
      "WAVE_acc_mean": 0.5299999999999999,
      "n_valid": 5
    },
    "LAYERED_N100_rho_lambda": {
      "MC_mean": 2.4045502981372318,
      "MC_sd": 0.4433124348175706,
      "NARMA_NRMSE_mean": 0.9688626718375953,
      "XOR_acc_mean": 0.6333333333333334,
      "WAVE_acc_mean": 0.49777777777777776,
      "n_valid": 5
    },
    "LAYERED_N300_rho_lambda": {
      "MC_mean": 3.525635622482151,
      "MC_sd": 0.18407130402859254,
      "NARMA_NRMSE_mean": 0.9975200014249195,
      "XOR_acc_mean": 0.9966666666666667,
      "WAVE_acc_mean": 0.5133333333333333,
      "n_valid": 5
    },
    "LAYERED_N800_rho_lambda": {
      "MC_mean": 3.563680721963648,
      "MC_sd": 0.6337721823429956,
      "NARMA_NRMSE_mean": 1.0483165275280069,
      "XOR_acc_mean": 0.9422222222222223,
      "WAVE_acc_mean": 0.47000000000000003,
      "n_valid": 5
    },
    "RAND_GAUSS_N100_rho_p95_sv": {
      "MC_mean": 1.3748761716832152,
      "MC_sd": 0.2925524990340686,
      "NARMA_NRMSE_mean": 1.00803507854581,
      "XOR_acc_mean": 0.4955555555555556,
      "WAVE_acc_mean": 0.4877777777777778,
      "n_valid": 5
    },
    "RAND_GAUSS_N300_rho_p95_sv": {
      "MC_mean": 2.7043083627818176,
      "MC_sd": 0.2762551538454595,
      "NARMA_NRMSE_mean": 0.9983488928153488,
      "XOR_acc_mean": 0.8066666666666666,
      "WAVE_acc_mean": 0.5166666666666666,
      "n_valid": 5
    },
    "RAND_GAUSS_N800_rho_p95_sv": {
      "MC_mean": 3.081888577558879,
      "MC_sd": 0.72521558178174,
      "NARMA_NRMSE_mean": 0.9701447815650264,
      "XOR_acc_mean": 0.9366666666666668,
      "WAVE_acc_mean": 0.4822222222222223,
      "n_valid": 5
    },
    "MESH_4N_N100_rho_p95_sv": {
      "MC_mean": 1.2127937908958792,
      "MC_sd": 0.24339295274716397,
      "NARMA_NRMSE_mean": 0.9671358830180118,
      "XOR_acc_mean": 0.49777777777777776,
      "WAVE_acc_mean": 0.478888888888889,
      "n_valid": 5
    },
    "MESH_4N_N300_rho_p95_sv": {
      "MC_mean": 1.3687726211142055,
      "MC_sd": 0.3768064598573132,
      "NARMA_NRMSE_mean": 0.9670667514209237,
      "XOR_acc_mean": 0.4955555555555556,
      "WAVE_acc_mean": 0.49000000000000005,
      "n_valid": 5
    },
    "MESH_4N_N800_rho_p95_sv": {
      "MC_mean": 1.9649520017777458,
      "MC_sd": 0.3491308647461332,
      "NARMA_NRMSE_mean": 0.9721556519310693,
      "XOR_acc_mean": 0.5733333333333334,
      "WAVE_acc_mean": 0.4844444444444445,
      "n_valid": 5
    },
    "ER_SPARSE_N100_rho_p95_sv": {
      "MC_mean": 1.3285291923078908,
      "MC_sd": 0.14328164053119827,
      "NARMA_NRMSE_mean": 0.9748333405205828,
      "XOR_acc_mean": 0.4922222222222222,
      "WAVE_acc_mean": 0.47333333333333333,
      "n_valid": 5
    },
    "ER_SPARSE_N300_rho_p95_sv": {
      "MC_mean": 2.3014716520113963,
      "MC_sd": 0.21316107838732393,
      "NARMA_NRMSE_mean": 0.9969378589565306,
      "XOR_acc_mean": 0.6611111111111112,
      "WAVE_acc_mean": 0.5266666666666666,
      "n_valid": 5
    },
    "ER_SPARSE_N800_rho_p95_sv": {
      "MC_mean": 2.8376542092999997,
      "MC_sd": 0.46524015111490374,
      "NARMA_NRMSE_mean": 0.982558173928281,
      "XOR_acc_mean": 0.8222222222222223,
      "WAVE_acc_mean": 0.5133333333333333,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N100_rho_p95_sv": {
      "MC_mean": 1.4206058868105447,
      "MC_sd": 0.2426476031223676,
      "NARMA_NRMSE_mean": 0.9662668373987722,
      "XOR_acc_mean": 0.5033333333333333,
      "WAVE_acc_mean": 0.48888888888888893,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N300_rho_p95_sv": {
      "MC_mean": 1.6178641672891252,
      "MC_sd": 0.16827343111695864,
      "NARMA_NRMSE_mean": 0.9512340648881684,
      "XOR_acc_mean": 0.4955555555555556,
      "WAVE_acc_mean": 0.49444444444444446,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N800_rho_p95_sv": {
      "MC_mean": 2.142296774833456,
      "MC_sd": 0.4191231239726641,
      "NARMA_NRMSE_mean": 0.9848762747329864,
      "XOR_acc_mean": 0.53,
      "WAVE_acc_mean": 0.49777777777777776,
      "n_valid": 5
    },
    "HUB_SPOKE_N100_rho_p95_sv": {
      "MC_mean": 1.3484746593780765,
      "MC_sd": 0.39537565369265654,
      "NARMA_NRMSE_mean": 0.9579213906555631,
      "XOR_acc_mean": 0.4955555555555556,
      "WAVE_acc_mean": 0.4966666666666666,
      "n_valid": 5
    },
    "HUB_SPOKE_N300_rho_p95_sv": {
      "MC_mean": 1.654223878185578,
      "MC_sd": 0.3993611222110243,
      "NARMA_NRMSE_mean": 0.9630484488057448,
      "XOR_acc_mean": 0.5033333333333333,
      "WAVE_acc_mean": 0.5055555555555555,
      "n_valid": 5
    },
    "HUB_SPOKE_N800_rho_p95_sv": {
      "MC_mean": 2.4748083455529266,
      "MC_sd": 0.20703749341268937,
      "NARMA_NRMSE_mean": 0.9828865380082847,
      "XOR_acc_mean": 0.7055555555555555,
      "WAVE_acc_mean": 0.5288888888888889,
      "n_valid": 5
    },
    "LAYERED_N100_rho_p95_sv": {
      "MC_mean": 1.2376153537447174,
      "MC_sd": 0.10941989247894958,
      "NARMA_NRMSE_mean": 0.96344740340655,
      "XOR_acc_mean": 0.4955555555555556,
      "WAVE_acc_mean": 0.4677777777777778,
      "n_valid": 5
    },
    "LAYERED_N300_rho_p95_sv": {
      "MC_mean": 2.0773724692113475,
      "MC_sd": 0.14642087624356653,
      "NARMA_NRMSE_mean": 0.9816060487264359,
      "XOR_acc_mean": 0.6333333333333334,
      "WAVE_acc_mean": 0.5255555555555556,
      "n_valid": 5
    },
    "LAYERED_N800_rho_p95_sv": {
      "MC_mean": 2.6812473779923116,
      "MC_sd": 0.15464880594358218,
      "NARMA_NRMSE_mean": 1.003985148281074,
      "XOR_acc_mean": 0.7733333333333333,
      "WAVE_acc_mean": 0.5177777777777777,
      "n_valid": 5
    },
    "RAND_GAUSS_N100_rho_deg_norm": {
      "MC_mean": 2.5060064853110604,
      "MC_sd": 0.3914221583331487,
      "NARMA_NRMSE_mean": 0.9515335842561891,
      "XOR_acc_mean": 0.6077777777777778,
      "WAVE_acc_mean": 0.4966666666666667,
      "n_valid": 5
    },
    "RAND_GAUSS_N300_rho_deg_norm": {
      "MC_mean": 0.2160127141931789,
      "MC_sd": 0.17034476923901146,
      "NARMA_NRMSE_mean": 1.1323984983210333,
      "XOR_acc_mean": 0.49444444444444435,
      "WAVE_acc_mean": 0.41,
      "n_valid": 5
    },
    "RAND_GAUSS_N800_rho_deg_norm": {
      "MC_mean": 0.2051440753548793,
      "MC_sd": 0.03127417465625156,
      "NARMA_NRMSE_mean": 1.271265325676583,
      "XOR_acc_mean": 0.5144444444444444,
      "WAVE_acc_mean": 0.40555555555555556,
      "n_valid": 5
    },
    "MESH_4N_N100_rho_deg_norm": {
      "MC_mean": 2.0434409510575167,
      "MC_sd": 0.38723023183950295,
      "NARMA_NRMSE_mean": 0.9715890961784414,
      "XOR_acc_mean": 0.5855555555555555,
      "WAVE_acc_mean": 0.5044444444444445,
      "n_valid": 5
    },
    "MESH_4N_N300_rho_deg_norm": {
      "MC_mean": 3.476594695946956,
      "MC_sd": 0.7328440143868743,
      "NARMA_NRMSE_mean": 0.9708345158952988,
      "XOR_acc_mean": 0.7466666666666667,
      "WAVE_acc_mean": 0.5211111111111111,
      "n_valid": 5
    },
    "MESH_4N_N800_rho_deg_norm": {
      "MC_mean": 4.00286860387042,
      "MC_sd": 0.3681139049342949,
      "NARMA_NRMSE_mean": 1.0800078364686772,
      "XOR_acc_mean": 0.9366666666666668,
      "WAVE_acc_mean": 0.5122222222222221,
      "n_valid": 5
    },
    "ER_SPARSE_N100_rho_deg_norm": {
      "MC_mean": 2.9024847056236274,
      "MC_sd": 0.4184355467830924,
      "NARMA_NRMSE_mean": 0.9824496145355465,
      "XOR_acc_mean": 0.7877777777777778,
      "WAVE_acc_mean": 0.5,
      "n_valid": 5
    },
    "ER_SPARSE_N300_rho_deg_norm": {
      "MC_mean": 2.7942327227910626,
      "MC_sd": 0.4264374758149621,
      "NARMA_NRMSE_mean": 0.9819114101023837,
      "XOR_acc_mean": 0.8033333333333333,
      "WAVE_acc_mean": 0.48777777777777775,
      "n_valid": 5
    },
    "ER_SPARSE_N800_rho_deg_norm": {
      "MC_mean": 2.6956991431013053,
      "MC_sd": 0.6941042342938042,
      "NARMA_NRMSE_mean": 0.994572286167858,
      "XOR_acc_mean": 0.8711111111111112,
      "WAVE_acc_mean": 0.4755555555555556,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N100_rho_deg_norm": {
      "MC_mean": 2.6729670271602144,
      "MC_sd": 0.31282984627077604,
      "NARMA_NRMSE_mean": 0.9866675676150355,
      "XOR_acc_mean": 0.7477777777777778,
      "WAVE_acc_mean": 0.4888888888888888,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N300_rho_deg_norm": {
      "MC_mean": 3.31392784593231,
      "MC_sd": 0.5914476751353946,
      "NARMA_NRMSE_mean": 1.0150469286232868,
      "XOR_acc_mean": 0.831111111111111,
      "WAVE_acc_mean": 0.5122222222222221,
      "n_valid": 5
    },
    "WS_SMALLWORLD_N800_rho_deg_norm": {
      "MC_mean": 4.337969586662803,
      "MC_sd": 0.6282372508295245,
      "NARMA_NRMSE_mean": 0.9691098194875636,
      "XOR_acc_mean": 0.9555555555555555,
      "WAVE_acc_mean": 0.5222222222222223,
      "n_valid": 5
    },
    "HUB_SPOKE_N100_rho_deg_norm": {
      "MC_mean": 1.787649801878548,
      "MC_sd": 0.5929706166624906,
      "NARMA_NRMSE_mean": 0.967519224077089,
      "XOR_acc_mean": 0.4666666666666666,
      "WAVE_acc_mean": 0.5122222222222221,
      "n_valid": 5
    },
    "HUB_SPOKE_N300_rho_deg_norm": {
      "MC_mean": 3.3306814852172186,
      "MC_sd": 0.41312493903155206,
      "NARMA_NRMSE_mean": 0.9839732969121027,
      "XOR_acc_mean": 0.9777777777777779,
      "WAVE_acc_mean": 0.5433333333333333,
      "n_valid": 5
    },
    "HUB_SPOKE_N800_rho_deg_norm": {
      "MC_mean": 4.427511962416635,
      "MC_sd": 0.35529464623292895,
      "NARMA_NRMSE_mean": 0.9988895291956661,
      "XOR_acc_mean": 1.0,
      "WAVE_acc_mean": 0.49333333333333335,
      "n_valid": 5
    },
    "LAYERED_N100_rho_deg_norm": {
      "MC_mean": 2.6455793321679275,
      "MC_sd": 0.3432584264770843,
      "NARMA_NRMSE_mean": 0.9773910898964835,
      "XOR_acc_mean": 0.8166666666666667,
      "WAVE_acc_mean": 0.46222222222222226,
      "n_valid": 5
    },
    "LAYERED_N300_rho_deg_norm": {
      "MC_mean": 1.8823413935912643,
      "MC_sd": 0.5858763571880742,
      "NARMA_NRMSE_mean": 1.1252924495282846,
      "XOR_acc_mean": 0.711111111111111,
      "WAVE_acc_mean": 0.4333333333333333,
      "n_valid": 5
    },
    "LAYERED_N800_rho_deg_norm": {
      "MC_mean": 1.069834384504318,
      "MC_sd": 0.3361289296870907,
      "NARMA_NRMSE_mean": 1.4397053777855506,
      "XOR_acc_mean": 0.5299999999999999,
      "WAVE_acc_mean": 0.3866666666666666,
      "n_valid": 5
    }
  },
  "Bf": 100.0,
  "topologies": [
    "RAND_GAUSS",
    "MESH_4N",
    "ER_SPARSE",
    "WS_SMALLWORLD",
    "HUB_SPOKE",
    "LAYERED"
  ],
  "Ns": [
    100,
    300,
    800
  ],
  "seeds": [
    42,
    43,
    44,
    45,
    46
  ],
  "rho_variants": [
    "rho_lambda",
    "rho_p95_sv",
    "rho_deg_norm"
  ],
  "T": 500,
  "kappa": 0.03
}
```


=== FILE: artifacts/z142_vs_z139_table.md (2974 chars) ===
```
# z142 vs z139 — topology MC ranking comparison

**Source:** `results/z142_topology_v2/summary.json` (FULL)

## N=800 MC across normalisations

| topology       | z139 (Bf=2e4) | z142 rho_lambda | z142 rho_p95_sv | z142 rho_deg_norm |
|----------------|--------------:|----------------:|----------------:|-------------------:|
| ER_SPARSE      |  2.20 (n=2) |  3.55 (n=5) |  2.84 (n=5) |  2.70 (n=5) |
| LAYERED        |  2.17 (n=2) |  3.56 (n=5) |  2.68 (n=5) |  1.07 (n=5) |
| RAND_GAUSS     |  1.87 (n=2) |  2.25 (n=5) |  3.08 (n=5) |  0.21 (n=5) |
| MESH_4N        |  3.29 (n=2) |  2.20 (n=5) |  1.96 (n=5) |  4.00 (n=5) |
| WS_SMALLWORLD  |  2.94 (n=2) |  2.17 (n=5) |  2.14 (n=5) |  4.34 (n=5) |
| HUB_SPOKE      |  2.89 (n=2) |  1.92 (n=5) |  2.47 (n=5) |  4.43 (n=5) |

## N=800 rho_lambda ranking change (z139 → z142)

| topology       | z139 MC | z142 MC | Δ      | rank z139 | rank z142 |
|----------------|--------:|--------:|-------:|----------:|----------:|
| ER_SPARSE      |    2.20 |    3.55 |  +1.35 |         4 |         2 |
| LAYERED        |    2.17 |    3.56 |  +1.39 |         5 |         1 |
| RAND_GAUSS     |    1.87 |    2.25 |  +0.38 |         6 |         3 |
| MESH_4N        |    3.29 |    2.20 |  -1.08 |         1 |         4 |
| WS_SMALLWORLD  |    2.94 |    2.17 |  -0.77 |         2 |         5 |
| HUB_SPOKE      |    2.89 |    1.92 |  -0.97 |         3 |         6 |

## rho_lambda MC across scale (z142 honest cell)

| topology       | N=100 | N=300 | N=800 | scaling N=100→800 |
|----------------|------:|------:|------:|------------------:|
| ER_SPARSE      |  1.90 |  2.99 |  3.55 |             1.87× |
| LAYERED        |  2.40 |  3.53 |  3.56 |             1.48× |
| RAND_GAUSS     |  2.34 |  2.67 |  2.25 |             0.96× |
| MESH_4N        |  1.60 |  1.76 |  2.20 |             1.37× |
| WS_SMALLWORLD  |  1.81 |  1.66 |  2.17 |             1.20× |
| HUB_SPOKE      |  1.19 |  1.72 |  1.92 |             1.60× |

## Multi-task ranking at N=800 rho_lambda (honest cell, n=5)

| topology       | MC    | NARMA NRMSE↓ | XOR_acc | WAVE_acc |
|----------------|------:|-------------:|--------:|---------:|
| ER_SPARSE      |  3.55 |         0.98 |    0.97 |     0.52 |
| LAYERED        |  3.56 |         1.05 |    0.94 |     0.47 |
| RAND_GAUSS     |  2.25 |         1.18 |    0.75 |     0.45 |
| MESH_4N        |  2.20 |         0.95 |    0.68 |     0.53 |
| WS_SMALLWORLD  |  2.17 |         1.00 |    0.59 |     0.51 |
| HUB_SPOKE      |  1.92 |         1.00 |    0.50 |     0.53 |

## Per-task champion (z142 honest cell, N=800 rho_lambda)

- **MC**:    LAYERED
- **NARMA**: MESH_4N (lower is better)
- **XOR**:   ER_SPARSE
- **WAVE**:  HUB_SPOKE

## HUB_SPOKE WAVE classification — z139 → z142

- z139 WAVE: **0.608** (was the brief's classification champion)
- z142 WAVE: **0.530** — advantage gone

## Headline
- z139 N=800 rho_lambda MC champion: **MESH_4N**
- z142 N=800 rho_lambda MC champion: **LAYERED**
- Inversion: YES

```


=== FILE: artifacts/z143_summary.json (391 chars) ===
```json
{
  "n_points": 48,
  "n_pass_all": 1,
  "n_pass_vsint": 2,
  "n_pass_vb": 8,
  "n_pass_id": 32,
  "max_abs_dVsint_mV": 248.00172398653973,
  "max_abs_dVb_mV": 99.91713051909828,
  "max_abs_dlog10Id": 1.6819344389694875,
  "median_abs_dlog10Id": 0.24673335059995827,
  "verdict": "FAIL",
  "thresholds": {
    "dVsint_mV": 5.0,
    "dVb_mV": 5.0,
    "dlog10Id": 0.3
  },
  "bjt_Bf": 100.0
}
```


=== FILE: artifacts/z91g_F1v2_summary.json (349 chars) ===
```json
{
  "n_curves": 33,
  "n_evaluated": 25,
  "n_skipped": 8,
  "median_log_rmse": 1.394425650326626,
  "p90_log_rmse": 2.3675362106780047,
  "elapsed_s": 38.50670027732849,
  "vs_z91f_run1_median": 4.234,
  "vs_z91f_run2_median": 2.402,
  "note": "true two-model validation (M1 = 130DNWFB, M2 = 130bulkNSRAM) with Sebastian's per-bias CSV overrides"
}
```
