# CAMPAIGN_FULL_PUSH v4 — 2026-05-19 → 2026-06-01 (13 days)

**Supersedes:** v3 (2026-05-19, morning). v3 kept "no Sebas wait" stance — v4 preserves that AND
absorbs today's three landed Pillar-I analyses + the C1 KILLSHOT + Pillar-III + Pillar-V results,
folds in O90 oracle 3/3 unanimous critique, and locks the brief deadline.

**Brief deadline:** 2026-06-01 (13 days). MARIO_BRIEF_v4.7 is the deliverable.

---

## What's new vs v3 (executive delta)

1. **C1 NPN excluded as candidate.** Pillar-I/C1 ran today: NPN-OFF improves median by 1.19 dec
   vs NPN-ON (3.27 vs 4.46), heatmap basin floor 2.1 dec, NPN-OFF killshot triggered.
   Caveat: torch `bsim4_port` already wires NPN parallel — so the 1.62 dec achievement is **with
   NPN ON**. Yesterday's "I_par=100 nA constant" finding is **not** the NPN. C1 is removed from
   the candidate list. C3 BSIM4 JTS-TAT promoted to top priority; C2 floating-body sub-Vt added;
   C4 measurement-artifact / contact-resistance hypothesis newly added.
2. **Pillar-I.1 verdict landed: MOSFET-LIKE.** 1/33 biases pass rectification gate. Caveat
   in verdict: protocol measures fwd/bwd sweep *hysteresis*, not true ±Vd rectification; still
   informative as prior narrowing.
3. **Pillar-I.3 verdict landed: well-tap best-fit by VG2 slope.** d log10|Id|/dVG2 ≈ −0.013 to
   −0.075 dec/V at VG1=0.6, all four Vd probes vote well-tap. R² weak (0.32–0.48).
4. **Pillar-I (T-scaling prediction): well-tap predicts 1.5e-20 A at the diagnostic 250 nA bias** —
   16 dec below measured. **All three standard candidates (junction, STI, GIDL) ruled out as
   primary conductor.** I.3's "well-tap best-fit by slope" is a residual-shape signature only;
   well-tap cannot be the dominant magnitude path.
5. **Pillar-III topology zoo: PASS both gates.** 104/120 edge-of-chaos; max NARMA-30 r² = 0.994
   (sine+noise, gameable); honest broadband-stim best NARMA = 0.968 (WS+Mackey-Glass, σ=0.10).
   Top combos: WS_SMALL_WORLD, HIERARCHICAL_3LEVEL, BA_SCALE_FREE on Mackey-Glass with σ ≥ 0.05 V.
6. **Pillar-V emulator: LOVO 3.06 dec FAIL extrapolation, curve-CV 0.119 dec PASS interpolation.**
   `predictor.pkl` shippable for VG1 ∈ [0.2, 0.6] **internal use only** (per O90: ML is
   engineering safety-net, not science).
7. **Pillar-IV.1 Verilog-A skeleton complete**, 26/26 begin/end balanced. **Caveat:** M1/M2
   intrinsic Ids uses textbook square-law (not full BSIM4). IV.2 33-bias ngspice validation in
   flight today.
8. **O90 3/3 unanimous critique absorbed**: I.6 promoted to HEADLINE deliverable. I.1/I.3
   reclassified prior-narrowing (not falsifying gates). I.4 explicitly tagged "model-derived
   prior, not independent evidence". ML emulator framing locked INTERNAL "safety-net".
   I.1-style killshots require: frozen ROI, block-bootstrap, symmetric 3-outcome gate.
9. **Geometry from Sebas .asc (I.2 verdict):** W=0.36 µm, L_M1=180 nm, L_M2=1800 nm, tox=4 nm.
   **Process node 130 vs 180 nm ambiguity flagged** — one-line confirmation request to Sebas.
10. **DC-fit honest baseline: 1.62 dec** (parallel-path I_par=100 nA constant, torch port, NPN
    ON, 33-bias grid, fwd+bwd). Not under 1.0 dec yet. Mechanism for the 100 nA still
    unidentified post-C1.

---

## Candidate shortlist (post-2026-05-19)

| ID | Candidate | Status | Owner |
|----|-----------|--------|-------|
| C1 | Parasitic NPN (Gummel-Poon parallel) | **DEAD** (killshot 2026-05-19) | — |
| C2 | Self-consistent floating-body M1 sub-Vt (body-bias raises subthreshold, settles at 100 nA-floor) | NEW backup | daedalus |
| C3 | BSIM4 JTS-TAT (trap-assisted-tunneling in junction, body-S/D) | TOP PRIORITY, in flight | daedalus |
| C4 | Measurement artifact / contact-resistance floor (~250 nA is plausible probe leakage at low Vd) | NEW probe | ikaros |
| C5 | (residual) GIDL/STI residual on top of dominant non-standard path | ruled out as primary by I.4 magnitude; kept as small-amplitude term | — |

---

## North star
Mario brief 2026-06-01 ships with:
1. **Honest DC story.** 1.62 dec on 33-bias grid is the current achievable. Sub-1.0 dec is a
   stretch goal contingent on C3 (JTS-TAT) landing or Sebas data unlocking T/W dimensions.
2. **Mechanism-of-100 nA candidate shortlist, not a specific identification.** Explicit
   caveat in brief.
3. **Validated transient** against ngspice synthetic (IV.2, in flight).
4. **Verilog-A shipped** (`nsram_cell_2T.va`) with documented square-law-vs-BSIM4 fidelity
   caveat.
5. **Pillar-III topology phase diagram** at scale on distilled BSIM4 cell.
6. **I.6 adversarial bias list** as the headline external-falsifier deliverable.
7. **predictor.pkl** as internal safety-net (NOT presented as a scientific claim).

## Operating principle
- NO waiting on Sebas. Internal-falsifier extraction first; Sebas reply uplifts I.4/I.6.
- All numbers **fwd+bwd, n=33-bias, bootstrap CI**. Pre-registered LOVO/LOGO before quoting.
- Multi-oracle (gpt5+gemini+grok) on EVERY pillar before declaring done.

---

## Pillar I — Internal physics discriminators (status update)

### I.1 Forward vs Backward Vd asymmetry  [DONE]
- Verdict: **MOSFET-LIKE** (1/33 biases pass; gate was 10/33 at |log10 R| ≥ 0.5, p<0.01).
- Reclassified per O90: **prior-narrowing, not falsifying gate**. Quoted in brief as such.
- Caveat acknowledged: sweep is fwd/bwd of |Vd|, not ±Vd polarity. **Strict-mode rerun**
  (frozen ROI + block-bootstrap, symmetric three-outcome gate {diode, mosfet, indeterminate})
  scheduled Day 2.

### I.2 Geometry extraction  [DONE]
- W=0.36 µm, L_M1=180 nm, L_M2=1800 nm, tox=4 nm, T=unannotated.
- **Process-node ambiguity (130 vs 180 nm) flagged** to Sebas as one-line follow-up.
- Single-geometry discriminator (well-tap ∝WL, STI ∝2(W+L), GIDL ∝W) NOT possible without
  ≥2 geometries. Pyport W/L sweep deferred.

### I.3 VG2 sensitivity at VG1=0.6  [DONE]
- Slopes −0.013 to −0.075 dec/V across 4 Vd probes → all vote well-tap.
- Reclassified per O90: **prior-narrowing, not gate**. Quoted as such in brief.
- Caveat: I_par residual not directly computable (no `I_model_no_par` in tree); used |Id| at
  low-Vd low-VG2 as proxy.

### I.4 Temperature scaling prediction  [DONE]
- Per-candidate predictions at the 250 nA diagnostic bias (300→400 K), all from BSIM4 model:
  - well-tap: 1.5e-20 A at 300 K (16 dec below measured); +5.17 dec to 400 K (Arrhenius).
  - STI: 0 A at 300 K (model not active at this regime).
  - GIDL: ~1e-37 A at 300 K; +0.82 dec to 400 K.
- **Reclassification per O90: model-derived PRIOR, not independent evidence.** Triangulates
  what to ask Sebas to measure; cannot stand alone.
- **Magnitude killshot**: all three standard candidates orthogonally undershoot the measured
  250 nA at the diagnostic bias by ≥16 dec. Whatever the 100 nA is, it isn't any of them.

### I.5 Pre-registered LOVO protocol  [PARTIAL]
- `PREREG_LOVO_VG1_GATED_BRANCH.md` not yet committed. Day 1–2 hard requirement.
- Per O90: must include **frozen ROI**, **block-bootstrap over fwd/bwd pairs** (not iid),
  and **symmetric three-outcome gate**. LOCK before any post-prereg fit number ships.

### I.6 Adversarial-prediction divergence — HEADLINE  [DONE-skeleton, ship as headline]
- `killshot_biases.json` landed: top-3 with max divergence 25.7–25.3 dec.
  - rank-1 (well-tap vs STI): VG1=0.0, VG2=−0.2, Vd=1.4 V, T=400 K.
  - rank-2 (well-tap vs GIDL): VG1=0.3, VG2=−0.2, Vd=1.4 V, T=400 K.
  - rank-3 (STI vs GIDL): VG1=0.0, VG2=0.6, Vd=0.40 V, T=220 K.
- **Promote to HEADLINE deliverable per O90.** Send to Sebas as adversarial bias list with
  pre-registered which-candidate-each-discriminates.
- Caveat: candidate set now narrowed since I.4 magnitude killshot — bias list is *still*
  useful because it discriminates *among* the priors even if none is the dominant path
  (residual-amplitude question vs primary-path question).

### I.7 (NEW) C3 BSIM4 JTS-TAT fit  [daedalus, in flight]
- Activate BSIM4 JSWS/JSWG TAT terms; sweep TAT activation params over physics-prior
  bracket; check whether constant 100 nA emerges from model at VG1=0.6/low-Vd.
- Gate: median dec ≤ 1.0 with JTS-TAT alone → C3 plausibility +1; else C3 not the answer.

### I.8 (NEW) C2 floating-body M1 sub-Vt self-consistent fit  [daedalus, backup]
- Re-solve Vbs self-consistent loop, allow M1 to operate moderately above sub-Vt at low VG1.
- Gate: same as I.7.

### I.9 (NEW) C4 measurement-artifact probe  [ikaros]
- Hypothesis: 250 nA constant residual is probe leakage / contact-resistance floor.
  - 250 nA / 50 mV = 5 MΩ — physically plausible for a low-current probe-pad leakage.
  - Check: is the "constant" really constant across VG1, VG2, Vd? Compute std/mean.
  - Check Sebas's instrument range — at Vd=0.05 V the current is at the 1-σ floor of many
    SMUs.
- This is a **PARSIMONY** check: before claiming new physics, exclude the artifact.

---

## Pillar II — Transient via ngspice synthetic
Unchanged from v3 (II.1 ngspice synthetic, II.2 small-signal AC, II.3 hysteresis-rate).
IV.2 33-bias validation pipes into II.1 ngspice deck construction.

---

## Pillar III — Network topology zoo  [DONE both gates]
- 104/120 edge-of-chaos PASS, max NARMA r² = 0.994 PASS, killshot not triggered.
- Top broadband-stim primitives (honest, sine+noise excluded as carrier-latching):
  - WS_SMALL_WORLD + Mackey-Glass, σ=0.10 → NARMA r² 0.968
  - HIERARCHICAL_3LEVEL + Mackey-Glass, σ=0.05 → 0.955
  - BA_SCALE_FREE + Mackey-Glass, σ=0.10 → 0.937
- σ ≥ 0.05 V heterogeneity required; homogeneous nodes globally-sync.
- Caveats (all explicit in verdict, repeated in brief): in-sample readout (no held-out
  train/test); Rosenstein-proxy Lyapunov; runs *inside* curve-CV interpolation envelope
  (val median |res| = 0.076 dec for distilled MLP), not at the 1.62 dec full-DC gate.
- Day 6–7: train/test split rerun to tighten NARMA upper-bound.

---

## Pillar IV — Verilog-A export
### IV.1  [SKELETON DONE]
- `nsram/verilog_a/nsram_cell_2T.va` written, 26/26 begin/end balanced, KCL signs match
  pyport residuals 1:1.
- **Fidelity caveat (locked in brief):** M1/M2 intrinsic Ids = textbook square-law +
  body-effect + CLM, NOT full BSIM4 (mobility/DIBL/CLM-regions/poly/quantum/moderate-inv).
  Expected mismatch 0.1–1 decade in Id; foundry BSIM4 VA can be dropped in by swapping
  the two Ids blocks (param names preserved).
### IV.2  [IN FLIGHT]
- ngspice harness pyport↔VA 33-bias diff today. Gate: max-RMSE ≤ 1e-4 (numerical agreement,
  not silicon agreement). >1e-4 → halt VA-shipped claim, document residual structure.
### IV.3  [Day 4]
- Cadence-Spectre lint.

---

## Pillar V — ML emulator (**reframed: internal engineering safety-net, NOT scientific claim**)
Per O90 unanimous critique: ML emulator is shippable engineering, not science.
- V.1 done: LOVO worst 3.06 dec FAIL extrapolation; curve-CV worst 0.119 dec PASS interpolation.
- `predictor.pkl` shippable for **internal use only** within VG1 ∈ [0.2, 0.6].
- Brief mentions it as a *deliverable* not a *result*. No SHAP-feature-importance is presented
  as physics insight (per O90 — SHAP on a 3-VG1-row dataset doesn't generalize).

---

## Pillar VI — Daily oracle 3-way (unchanged from v3)
- 09:00, 14:00, 19:00, 02:00 local. gpt5+gemini+grok read 01_LOG.md tail + newest pillar
  artifact. Q1 fragility, Q2 best-falsifier, Q3 NO-CHEAT drift. ≥2/3 SAME → ALERT.

---

## 13-day Gantt (locked)

| Day | Date | ikaros | daedalus | zgx |
|-----|------|--------|----------|-----|
| 1 | 05-19 (today) | I.5 LOVO pre-reg DRAFT + I.9 C4 artifact probe + brief v4.7 outline | I.7 C3 JTS-TAT dispatch + IV.2 ngspice 33-bias diff | III rerun w/ train-test split |
| 2 | 05-20 | I.5 LOVO LOCK + I.1 strict-mode rerun (frozen ROI + block-bootstrap) | I.8 C2 floating-body self-consistent | II.1 ngspice 7-rate transient kickoff |
| 3 | 05-21 | C4 artifact verdict + I.6 packaged-for-Sebas | I.7/I.8 verdicts | II.2 small-signal AC |
| 4 | 05-22 | brief v4.7 §1-3 draft | IV.3 Cadence lint | II.3 hysteresis-rate |
| 5 | 05-23 | brief v4.7 §4-5 draft | C3/C2 fit rerun w/ LOVO | III phase-diagram tightening |
| 6 | 05-24 | mid-campaign 3-way oracle | weekly synth | weekly synth |
| 7 | 05-25 | brief v4.7 §6-7 draft + figures | predictor.pkl packaging + caveat doc | edge-cases |
| 8 | 05-26 | full-brief v4.7 internal review draft | VA Cadence-port stress-test | adversarial scaling N=4096 |
| 9 | 05-27 | revisions from O91 critique | tape-out cell-card v3 | weekly synth |
| 10 | 05-28 | revisions cont. | predictor.pkl frozen | full integration |
| 11 | 05-29 | brief v4.7 final pre-read | final fit table | final synth |
| 12 | 05-30 | brief v4.7 LOCK + Mario one-pager | final tape-out card | final figs |
| 13 | 05-31 | commit, send-ready brief packet | sign-off | sign-off |

**Slip-buffer:** 06-01 = send day.

---

## Pre-registered killshots (frozen 2026-05-19)
- **I.1 strict-mode**: if frozen-ROI block-bootstrap |log10 R| < 0.2 on all 33 → diode dead.
  Current data: 1/33 pass at 0.5 threshold. Already in "MOSFET-like" basin.
- **I.4 T-scaling** (Sebas data dependent): ΔI/decade outside [Arrhenius±0.5, weak±0.2] → ALL
  three standard candidates wrong → publish missing-physics note.
- **I.6 adversarial** (Sebas data dependent): if a single bias outside the in-distribution
  grid shows >2 dec ANY model error → 2-of-3 candidates falsified.
- **I.7 C3 JTS-TAT**: ≤1.0 dec → C3 plausible.
- **I.8 C2 floating-body**: ≤1.0 dec → C2 plausible. Both >1.0 dec AND I.9 C4 also rules
  artifact out → publish honest "mechanism unidentified" note.
- **I.9 C4 artifact**: std(I_par)/mean(I_par) < 0.05 over 33 biases AND scales with probe
  pad geometry estimate → artifact plausible, retract physical-mechanism framing.
- **II.1 ngspice**: pyport vs ngspice NRMSE > 20% → solver bug, halt transient claims.
- **III.4 phase-diagram (re-gate w/ train-test split)**: max NARMA r² < 0.5 held-out → cell
  surrogate too smooth, redo distillation.
- **IV.2 VA-diff**: max-RMSE > 1e-4 → VA bug, halt VA-shipped claim.

---

## Acceptance criteria Day 13
**WIN (≥3 of):** DC ≤ 1.0 dec at honest LOVO; ngspice-validated transient; VA shipped+lint-clean;
predictor.pkl frozen+caveat-doc; topology-zoo train-test-split phase diagram. → Methods+device-claims
paper.
**PARTIAL (2):** Methods paper only. **PASS** for Mario brief.
**FAIL (≤1):** Honest-negative-result note + I.6 adversarial bias list to Sebas. Brief still ships
but as "what we measured, why we can't yet identify mechanism".

---

## Why this differs from v3
- v3 listed C1 NPN as a live candidate; v4 marks it DEAD (killshot 2026-05-19).
- v3 had I.1/I.3 as gates; v4 reclassifies them prior-narrowing per O90.
- v3 had I.4 as evidence; v4 demotes to model-derived prior per O90.
- v3 had I.6 as adversarial divergence sweep; v4 promotes to HEADLINE per O90.
- v3 had ML emulator as "brief safety-net"; v4 hard-locks INTERNAL ONLY, removes from
  scientific-claim slot.
- v3 had no C2/C3/C4 explicit candidates; v4 lists them as the candidate shortlist replacing
  the dead C1.
- v3 had no I.1 strict-mode (frozen ROI / block-bootstrap / symmetric gate); v4 adds.
- v3 LOVO pre-reg was Day 2; v4 hardens it Day 1–2 with O90 stipulations and makes it
  the gate before any new "best dec" number ships.
- v3 had geometry as TODO; v4 has W=0.36 µm/L_M1=180 nm extracted + process-node ambiguity
  flagged for Sebas confirmation.
- v3 9 Pillars; v4 same scaffold but I.7/I.8/I.9 added and Pillar-V demoted.
