# POST-AUDIT FIX PLAN — 2026-05-11

**Trigger**: Exhaustive review of Mario+Sebas-authored slides (49 images,
Zoom transcript) surfaced structural gaps in modelling and network experiments
that weren't visible from the email thread alone.

**Status of prior plans**:
- VG2_CONTINUUM_PLAN.md — CLOSED (both steps failed honestly)
- NEXT_DIRECTION_PLAN.md (NS-RAM vs ESN matrix) — CLOSED (14 cells, 0 wins)
- This plan SUPERSEDES the idle-correct posture from O41 oracle review.
  O41 was right *within* the reservoir-vs-ESN framing; this plan addresses
  the framing itself.

**NO-CHEAT PRINCIPLE STILL APPLIES (verbatim)**:
1. Never bend pre-registered acceptance gates post-hoc.
2. n ≥ 5 seeds per network condition, mean+std+CI bars, no single-seed
   pilots in brief writeups.
3. If a gate fails, log honestly and decide whether to pre-register a
   corrected gate before re-running.

**Brief v4.3 stays locked** while this work proceeds. If any step crosses
its gate, we open a v4.4 conversation explicitly. None of these steps
backdate prior null results — they extend coverage.

---

## Phase M — Modelling fixes (pyport)

### M1. PWL(V_G2) bulk-current refactor [HIGH]

**Source slide**: 12.26 (Sebas's "Semi-empirical model fits for impact
ionization bulk currents").

**Functional form Sebas uses**:
```
I_exp = 10^(d · V_d)
I_pwl = a · V_d^c + b      for V_d ≥ −j
      = 0                   for V_d <  −j
```
with `a, b, c, d` as PWL(V_G2), `j` constant.

**What pyport does**: polynomial(V_G1, V_G2) on top of BSIM4 §6.1.

**Implementation**:
- Add `PWLBulkCurrentParams` dataclass mirroring Sebas's form.
- Knot points: V_G2 ∈ {−0.20, −0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50}
  (matches Sebas's measurement coverage).
- Refit on 33-bias DC family with same Bf=9000, V_a=0.55, I_s=1e-9 anchor.

**Pre-registered gate**: PWL fit residual ≤ poly fit residual − 0.05 dec
on the 33-bias regression target. PASS = strict improvement.

**Failure path**: if PWL ≈ poly within ±0.05 dec, log "functional form is
not the binding constraint, polynomial substitutes adequately"; do NOT
fold into the brief.

**Wall budget**: 2h (1h implement, 1h fit + diagnostic).

---

### M2. Per-V_G1-branch residual decomposition [HIGH]

**Source slide**: 13.24 (Sebas's composite 4-parameter plot — three
explicit branches indexed by V_G1 ∈ {0.20, 0.40, 0.60}).

**Hypothesis**: our unified poly(V_G1, V_G2) may average across the three
V_G1 branches, hiding regime-specific structure.

**Implementation**:
- Compute pyport DC fit residual per (V_G1, V_G2) bin.
- Group by V_G1 ∈ {0.20, 0.40, 0.60}.
- Report mean residual per branch + bootstrap 95% CI.

**Pre-registered gate**: PASS if any single branch shows residual ≥ 0.30
dec above the cross-branch average → indicates branch-specific fit is
warranted. FAIL otherwise → unified poly is adequate.

**Wall budget**: 1h (analysis only, no re-fitting).

---

### M3. NFACTOR_M2 unclamped refit [MEDIUM]

**Source slide**: 13.24 (NFACTOR(V_G2) red branch reaches ~12 at
V_G2 = −0.2 V).

**Problem**: pyport BBO fit has NFACTOR_M2 bound at 3.0; that's an order
of magnitude below the silicon range.

**Implementation**:
- Re-run BBO with NFACTOR_M2 ∈ [3, 15].
- Compare final loss vs current 3.0-clamped fit.

**Pre-registered gate**: PASS if unclamped final loss < clamped final
loss − 0.05 dec. FAIL otherwise → silicon range matters but doesn't bind
the fit.

**Wall budget**: 1h.

---

### M4. Negative-V_G2 leakage suppression probe [MEDIUM]

**Source slide**: 12.30 (V_G ≈ −2 V on second transistor used to
"drive down the leaky behaviour").

**Problem**: our DC fit window is V_G2 ∈ [−0.2, 0.5] V. We've never
modelled the −2 V regime.

**Implementation**:
- Run pyport DC at V_G2 = −2 V across V_G1 ∈ {0.2, 0.4, 0.6} V.
- Compare leakage current to the off-state level Sebas reports
  visible in slide 12.30 (need to extract numeric from image — flag
  this as a measurement-uncertainty input).

**Pre-registered gate**: PASS if simulated I_d at V_G2=−2 V is within
1 decade of the slide-extracted measurement. FAIL → physics gap;
document.

**Wall budget**: 1h.

---

### M5. 1T deep-Nwell floating-body variant scaffold [LOW]

**Source slide**: 12.29(1) (1T, 8 µm², 180 nm CMOS).

**What this gives us**: a second cell-family member to compare against.
Mario explicitly markets both 1T and 2T variants.

**Implementation**:
- Add `OneTFloatingBodyCell` to pyport (geometry + BSIM4 1T + body diode).
- No fit — just produce a single I-V family at V_G1 ∈ {0.2, 0.4, 0.6} V.

**Pre-registered gate**: shape-of-curve sanity only. No quantitative gate.

**Wall budget**: 2h. Lowest priority. Skip if compute budget tight.

---

## Phase N — Network experiments (the structural one)

### N1. Brian2-style LIF SNN baseline reproduction [CRITICAL]

**Source slide**: 12.33 + 12.39 (Sebas's existing SNN benchmark:
LIF 72% vs Poisson reference 85%).

**Goal**: reproduce the 72% LIF accuracy from his Brian2 setup with our
PyTorch implementation, before any NS-RAM substitution.

**Implementation**:
- LIF model: exponential decay with parameters from slide 12.33
  (TIMESCALE_DIV=1e-3, V_TH=0.4–0.5, V_RESET=−0.84, EXC_GAIN=−0.6).
- Time rescaling 10³× from CMOS for stability (matches Sebas).
- Training: Poisson rate-encoded input on MNIST or simpler digit set.
- Inference: LIF with same trained weights.
- n=5 seeds, bootstrap CI.

**Pre-registered gate**: PASS if mean accuracy ∈ [70%, 74%] across 5
seeds (i.e., reproduces Sebas's 72% within ±2 pp). FAIL → setup
mismatch; needs Sebas's script.

**Wall budget**: 4h.

**Dependency**: this step is INDEPENDENT of Sebas (PyTorch implementation
from scratch). Comparison to Sebas's actual code is a separate cross-check
when his script arrives.

---

### N2. NS-RAM surrogate substituted as LIF input neuron [CRITICAL]

**Source slide**: 13.33 (NSRAM firing as input-neuron soma with V_G2
linear range 2.5–3.0 V driving 1 V output pulses).

**Goal**: replace plain LIF input neurons with NS-RAM 4D body-state
surrogate operating as input neurons. Measure SNN accuracy delta.

**Implementation**:
- Use the existing 4D surrogate (V_G1, V_G2, V_d, V_b) from
  `results/z220_4d_dense/surrogate_4d_dense.npz`.
- Mode: input neuron (drive V_G1 with input current, read out spike
  events on V_d threshold).
- All other layers identical to N1.
- n=5 seeds, same train/test split as N1.

**Pre-registered gate**:
- PRIMARY: PASS if NS-RAM-input mean ≥ plain-LIF mean − 2 pp with
  non-overlapping CIs (i.e., not strictly worse).
- AMBITIOUS: PASS if NS-RAM-input mean > plain-LIF mean with
  non-overlapping CIs.

**Failure path**: if NS-RAM substitution degrades > 2 pp, log honestly
and probe whether the surrogate's time-constants need re-fitting to
Sebas's transient data (which requires data arrival).

**Wall budget**: 3h after N1.

---

### N3. Time-rescaling sanity sweep [MEDIUM]

**Source slide**: 12.33 ("Timescale is slowed down (10³) from the
NS-RAM CMOS simulations for ease of Brian2 convergence").

**Goal**: confirm 10³× rescaling reproduces silicon behaviour without
artefact.

**Implementation**:
- Run N2 at three rescalings: 10², 10³, 10⁴.
- Compare accuracy, firing rate, mean spike interval.

**Pre-registered gate**: PASS if accuracy at 10³× matches accuracy at
10² within 2 pp (i.e., rescaling is benign).

**Wall budget**: 2h.

---

### N4. Mirror-bank weight encoding [MEDIUM]

**Source slide**: 13.33 ("Weight comes from the mirror bank generating
the bias voltage").

**Goal**: encode network weights as V_G2 bias voltages rather than as
multiplicative scalars. Tests the architectural claim that NS-RAM weights
ARE V_G2.

**Implementation**:
- Map weight w ∈ [−1, +1] → V_G2 ∈ [linear range upper, lower].
- For thick-ox: V_G2 ∈ [2.5, 3.0] V.
- For thin-ox: V_G2 ∈ [0.0, 0.5] V (the regime we've actually fit).
- Re-run N1/N2 with this weight encoding.

**Pre-registered gate**: PASS if mirror-bank-weight accuracy ≥ uniform-
weight baseline within ±2 pp. Architectural validation, not a win
condition.

**Wall budget**: 3h after N2.

---

### N5. Soma → mirror-bank → storage cell chain [LOW-MEDIUM]

**Source slide**: 13.31 (self-reset integration cell) + 13.33 (full
chain).

**Goal**: simulate the full input-neuron → weight bank → storage
chain as a 3-cell circuit, not as 3 abstract operations.

**Implementation**:
- Compose: NS-RAM input neuron (thick-ox) → V_G2 weight bank (mirror)
  → NS-RAM storage cell (thin-ox).
- Drive with single Poisson rate, measure output.
- Compare reproduced spike train against slide 13.31.

**Pre-registered gate**: PASS if simulated spike-train pattern matches
slide 13.31 qualitatively (inter-spike interval, peak height, decay
within ±20%).

**Wall budget**: 3h after N4. Lowest priority of N phase.

---

## Phase A — Asks (already drafted)

In `consolidated_email_2026-05-11_draft.md`:
- (a) B_f single-bias I_c/I_b
- (b) 7-rate τ spectrum
- (c) Thick-ox cell card
- (d) Repository link
- (e) NFACTOR_M2 realistic range
- (f) High-V_G2 / high-V_G1 coverage-gap sweeps
- (g) Experiment list 3–7 (named)
- (h) Bulk-current PWL coefficient table
- (i) Brian2 SNN benchmark script + dataset

These DON'T need our compute time. They're queued for user-side send.

---

## Sequencing + dependencies

```
M1 PWL refactor ─┐
M2 branch decomp ┤
M3 NFACTOR unclamp ┼─── can run in parallel (CPU-light, independent)
M4 −2V probe ─────┤
M5 1T variant ────┘

N1 LIF baseline ──→ N2 NS-RAM input neuron ──→ N3 timescale
                                              ──→ N4 mirror-bank weights
                                                      ──→ N5 full chain
```

Total wall: ~22h compute work, spread across cron-wake-up windows.

---

## Cron schedule (autonomous progression)

Old V_G2/ESN-matrix wake-ups will be retired (those plans are CLOSED).
Replacement schedule:

| Cadence | Purpose |
|---|---|
| `13 9,13,17,21 * * *` | POST_AUDIT_FIX wake-up — pick next pending M/N step |
| `47 * * * *` | hourly idle check-in (stretched from `:17` per O41) — moved to `:47` for spacing |
| `33 11,23 * * *` | 12h oracle review (kept) |
| `43 4 * * *` | baseline watchdog (kept) |
| `23 6 * * *` | morning brief (kept) |
| `13 2 * * *` | daily synthesis (kept) |
| `7 0 * * *` | resource audit (kept) |
| `23 9 * * 1` | weekly review Monday (kept) |
| `11 9,15,21 * * *` | 6h track audit (kept) |
| `23 3 * * *` | GPU off-hours (kept) |

The 2h POST_AUDIT_FIX cadence is the primary work-driver. Hourly idle is
the safety net (logs APU temp + sentinel state, picks up nothing if a
sequence is already active).

---

## Failure-honest exit conditions

This plan reaches "CLOSED — null" if:
- All M-phase gates fail (pyport fits don't improve with PWL / branch /
  NFACTOR unclamping) AND
- N-phase gate N2 fails strictly (NS-RAM-as-LIF degrades > 2 pp vs plain
  LIF baseline).

In that case: log explicitly, leave brief v4.3 final, retire this plan,
and the project is genuinely compute-closed (this time on the right
task class).

This plan reaches "BRIEF v4.4" if:
- N2 passes ambitious gate (NS-RAM strictly beats LIF in SNN) OR
- M1 + M2 jointly resolve the 0.51 dec residual to ≤ 0.30 dec.

Either outcome justifies an Overleaf revision.
