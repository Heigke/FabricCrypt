# CAMPAIGN_FULL_PUSH v3 — 2026-05-19 → 2026-06-01 (13 days)

**Supersedes:** v2 (2026-05-18). v2 was gated on Sebas reply. v3 explicitly removes that gate — we extract maximum signal from the data we already have and *build deliverables now*.

## North star
Mario brief 2026-06-01 must ship with: (1) honest DC story ≤1 dec if attainable, ≥1.6 dec if not, (2) validated transient against ngspice synthetic, (3) Verilog-A export usable in Cadence, (4) topology-zoo network results at scale on actual BSIM4 cell.

## Operating principle
- NO waiting on Sebas. Every hypothesis must have an *internal* falsifier extractable from data already in `data/sebas_2026_04_22/` + `data/sebas_2026_05_02/` + `docs/Zoom/` + `data/mario_slide*.json`.
- All numbers fwd+bwd, n=33-bias, with bootstrap CI.
- Pre-registered LOVO/LOGO before *any* fit result quoted.
- Multi-oracle (gpt5+gemini+grok) on EVERY pillar before declaring done.

---

## Pillar I — Internal physics discriminators (no Sebas dependency)
**Goal:** Decide well-tap vs STI vs GIDL using only what we have.

### I.1 Forward vs Backward Vd asymmetry  [ikaros]
- Existing 33-curve fwd+bwd already measured. Rectification = diode-like physics (well-tap junction or STI substrate diode), symmetric = MOSFET-like (GIDL, channel parasitic).
- Compute per-bias rectification ratio R(Vd) = I_fwd(Vd) / I_bwd(-Vd) where -Vd point in backward sweep exists.
- **Pre-registered gate**: |log10 R| ≥ 0.5 at ≥10/33 biases → diode-like, p<0.01 bootstrap → triggers well-tap branch.

### I.2 W/L extraction from Zoom slides  [ikaros]
- 27 jpegs in `docs/Zoom/` — re-OCR / vision-extract for any "W=" or "L=" annotations.
- Mario 0429 pptx — re-parse for transistor sizes.
- Sebas 2tnsram_simple.asc — pull explicit nmos4 sizes if disclosed.
- If found: predict junction-scaling (∝WL) vs STI-scaling (∝2(W+L)) vs GIDL (∝W) and check against any size-variant data.

### I.3 VG2 sensitivity probe at VG1=0.6  [ikaros]
- Existing 33-bias grid: filter to VG1=0.6 rows, plot I_par vs VG2 implied residual.
- Well-tap: insensitive to VG2 (body bias only)
- STI edge-FET: strong VG2 sensitivity (back-gate effect)
- GIDL: positive correlation with |VG-Vd|, sublinear in VG2

### I.4 Temperature scaling — extract from BSIM4 T-dependence  [daedalus]
- We have the model. Run pyport at T ∈ {220, 250, 300, 350, 400} K and predict what each candidate produces.
- Junction current: ∝ ni²(T) ≈ exp(-Eg/kT), strong Arrhenius
- STI edge-FET: weak T (mostly through u0(T))
- GIDL: weak T except tunnel-tail at high field
- Predicted slopes give a *prior* for the temperature-discriminator. When Sebas eventually replies, we already know what to look for. If Mario slides have ANY T-annotated curve, use as immediate cross-check.

### I.5 Pre-registered LOVO protocol  [ikaros]
- Write `research_plan/PREREG_LOVO_VG1_GATED_BRANCH.md`:
  - Leave-one-VG1-out folds (3 groups: VG1=0.2, 0.4, 0.6)
  - Physics priors: n_par ∈ [1.0, 3.0], I0 ∈ [1, 1000] nA, VG1_th ∈ [0.25, 0.45] V
  - Fwd+bwd parity max 0.3 dec spread else FAIL
  - Decade floor 2.907 dec (Vbs-clamped baseline) — held-out median must beat
  - BIC penalty ~10 nats before quoting "new best"
- LOCK before any fit number is reported in brief.

### I.6 Falsifier: adversarial-prediction divergence  [zgx]
- For each (well_tap, sti_edge_fet, gidl) candidate model with reasonable defaults, generate predicted I-V across an extended bias range (incl negative-Vd, T-sweep, W-sweep simulations).
- Find the bias condition (VG1*, VG2*, Vd*, T*) where pairwise prediction divergence is *maximum* (>2 dec ideally).
- If/when Sebas data lands, that single point falsifies 2-of-3.

---

## Pillar II — Transient via ngspice synthetic + small-signal AC
**Goal:** Validate transient solver without Sebas data.

### II.1 ngspice synthetic ground truth  [daedalus]
- Build 2T cell SPICE deck (nmos4 + parasitic diodes + VG2 ramp).
- 7-rate transient sweep: dV/dt ∈ {1V/10ns, 1V/100ns, 1V/1µs, 1V/10µs, 1V/100µs, 1V/1ms, 1V/10ms}.
- Compare against pyport transient_real_v2.py output.
- **Gate**: NRMSE ≤ 5% on V_B(t) across 7 rates → pyport validated against industry-standard tool.

### II.2 Small-signal AC from DC operating points  [ikaros]
- At each of 33 DC bias points, linearize Jacobian → get f_3dB, gain, phase. Predict transient step response from AC.
- Compare predicted step against pyport direct transient. Self-consistency check at zero cost.

### II.3 Hysteresis-rate-dependence  [daedalus]
- We already have fwd+bwd 33-curve sweep. The fwd-vs-bwd offset *is* a transient signature. Compute hysteresis area H(VG1,VG2) and check it scales with implied sweep rate.
- Sebas's measurement protocol may be inferable from this.

---

## Pillar III — Network topology zoo at scale
**Goal:** Use real BSIM4 cell (not FHN), explore non-trivial topologies, *learn structure* even with imperfect cell fit.

### III.1 Drop FHN, instantiate real BSIM4 cell in network nodes  [zgx]
- Surrogate-table version (current pyport too slow at N=1024): build a fast neural-net surrogate of pyport DC (regression NN on 24K bias points).
- Wire surrogate cells into network on zgx GB10.
- Topologies: ER_SPARSE, ER_DENSE, WS_small_world, BA_scale_free, MODULAR_4block, HIERARCHICAL_3level

### III.2 Heterogeneous parameter draws  [zgx]
- Sample 4096 cell instances from HMC posterior (when ready) — *or* from Gaussian-bracketed manufacturing variation if HMC still cooking.
- Drives diverge → break the trivial sync seen with uniform drive.

### III.3 Non-uniform input regimes  [zgx]
- Drive with: white noise, 1/f noise, sinusoid + noise, Poisson spike train, Mackey-Glass chaos
- Measure: Lyapunov spectrum, edge-of-chaos location, NARMA-30, Mutual Information, synergy (Φ_R, O_information)

### III.4 Topology × stimulus × heterogeneity 3D phase diagram  [zgx]
- 6 topologies × 5 stimuli × 4 heterogeneity levels = 120 cells (≈ 4 GPU-hr at N=4096)
- Output: which combinations cross edge-of-chaos? Which produce useful neuromorphic primitives?

---

## Pillar IV — Verilog-A export NOW
**Goal:** Mario+Sebas can run our model in Cadence/ngspice within days, not at the end.

### IV.1 Pyport → VA generator  [ikaros]
- Walk pyport AST: for each Python function representing a current/diode/cap, emit VA primitive.
- Use existing test_2t_homotopy as validation oracle.

### IV.2 ngspice harness  [daedalus]
- Generate ngspice deck instantiating the VA cell, sweep 33 biases.
- Compare against pyport DC output: pass if max-bias-RMSE ≤ 1e-6 (numerical agreement, not silicon agreement).

### IV.3 Cadence-compatibility lint  [ikaros]
- Run `va2cml` or manual lint for Cadence-Spectre compatibility (some VA features not supported).
- Document warnings in delivered package.

---

## Pillar V — Innovation: ML emulator + SHAP physics
**Goal:** Bypass model entirely for the practical deliverable AND learn what features matter.

### V.1 XGBoost + LightGBM + MLP emulator on Mario data  [daedalus]
- Train 3 models on existing 33-bias grid + Mario slide-extracted curves
- 60/40 stratified by VG1, 5-fold CV
- **Gate**: held-out test median ≤ 0.5 dec — *this is the brief's safety-net deliverable*

### V.2 SHAP feature importance  [daedalus]
- What input features (VG1, VG2, Vd, V_B-implied) drive prediction?
- Permutation importance on architectural features (channel-vs-parallel-path contributions)
- This tells us *what the data really wants to fit* independent of physics priors

### V.3 Ensemble with physics-aware regularization  [daedalus]
- Penalize predictions that violate Vth monotonicity, current-symmetry under bias-reversal, asymptotic decade-slope
- "Physics-informed ML" wins both audiences

---

## Pillar VI — Daily oracle 3-way (5h cadence)
- 09:00, 14:00, 19:00, 02:00 local
- Each cycle: gpt5+gemini+grok read 01_LOG.md tail + newest pillar artifact
- Q1 fragility, Q2 best-falsifier, Q3 NO-CHEAT drift
- ≥2/3 SAME flag → log corrective ALERT + halt that pillar pending fix

---

## Daily Gantt

| Day | ikaros | daedalus | zgx |
|-----|--------|----------|-----|
| 1 (today) | I.1 fwd-bwd rectification + I.2 zoom OCR + IV.1 VA gen scaffold | II.1 ngspice 7-rate transient + V.1 ML emulator | I.6 adversarial discriminator + III.1 cell surrogate NN |
| 2 | I.3 VG2 sens + LOVO pre-reg | I.4 T-scaling pred + V.2 SHAP | III.2 HMC posterior draws + III.3 topology start |
| 3 | IV.2 ngspice VA validation | II.2 small-signal AC + II.3 hysteresis-rate | III.3 stim-zoo |
| 4 | IV.3 Cadence lint + brief v4.7 outline | V.3 phys-aware ML | III.4 3D phase diagram |
| 5 | LOVO fit on validated branch | Oracle 3-way + integration | III.4 cont |
| 6 | brief v4.7 draft | tape-out cell-card v3 | III.4 finish + analysis |
| 7 | Mid-campaign synthesis + 3-way oracle | weekly synth | weekly synth |
| 8-10 | Refine + corrective falsifiers from oracle | Mario one-pager v2 | edge-cases + neuromorphic primitives |
| 11-12 | Brief v4.7 final + figures | predictor.pkl + VA.va shipped | full integration test |
| 13 | Commit. | Final synth. | Final synth. |

---

## Pre-registered killshots
- **I.1**: |log10 R| < 0.2 on all 33 → diode hypothesis dead, MOSFET-like favored
- **I.4 T-scaling**: ΔI per decade of T outside [Arrhenius±0.5, weak±0.2] → all candidates wrong, missing physics
- **II.1 ngspice**: pyport-vs-ngspice NRMSE >20% → solver has hidden bug, halt all transient claims
- **III.4**: All topologies trivial (R>0.95) or all chaotic (R<0.1) → cell surrogate too smooth, redo
- **IV.2**: max RMSE >1e-4 → VA bug, halt VA-shipped claim
- **V.1**: ML held-out >0.5 dec → emulator not deployable, fallback to honest negative-result paper

---

## Acceptance Day 13
**WIN**: ≥3 of {DC ≤1 dec, ngspice-validated transient, VA shipped, ML emulator ≤0.5 dec, topology-zoo phase-diagram} done. → Methods+device-claims paper.
**PARTIAL**: 2 done. → Methods paper only. **PASS** for Mario brief.
**FAIL**: ≤1 done. → Honest-negative-result note. Email Sebas asking for everything we couldn't extract.

---

## Why this differs from v2
- v2 had Pillar A gated on Sebas reply → dropped; replaced with internal-discriminator Pillar I
- v2 had Pillar F deliverables late → moved Verilog-A to Day 1-4 (critical path)
- v2 had emergent network dynamics as "exploratory" → now Pillar III with real cell + 3D phase diagram
- v2 didn't address Mario *deliverables shape*; v3 makes ML emulator a brief safety-net
- v2 had weekly oracle; v3 has 4x/day oracle on each pillar — kills cherry-picks fast
