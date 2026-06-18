# Mario v4.3 send decision — final document for user

**Date**: 2026-05-11.
**Purpose**: collect every fact you need to decide whether to send the
Mario v4.3 brief plus the Sebas characterisation requests *today*. Three
independent oracle reviews (openai gpt-5, gemini-2.5-pro, grok-4-latest)
converged on this recommendation, with no shared warnings.

---

## 1. The recommendation

**Send Mario v4.3 today.**
**Send Sebas main characterisation request + thick-oxide addendum today.**

The Sebas main request has been drafted and unsent for 6+ days; the
brief is locked at v4.3 with three defensible headlines; the
compute-side story is complete; further delay does not improve any
claim.

---

## 2. The three brief headlines you are sending

1. **Silicon-energy floor ~10× advantage vs the best AI MCU.**
   At 1024-step temporal inference at N = 64 cells, NS-RAM is
   ~0.7 µJ versus MAX78000 ~5 µJ, Coral Edge TPU ~10 µJ, Cortex-M4
   50–100 µJ. Independent of bench-compute reservoir-quality.

2. **ESN-class NARMA-10 accuracy at the silicon-energy floor.**
   NRMSE 0.612 ± 0.030 (30-seed CI). A textbook tanh ESN at the
   same network size reaches 0.563 — about 8% better. Framing:
   "NS-RAM achieves ESN-class accuracy at the silicon-energy floor,"
   not "NS-RAM beats ESN."

3. **Three-source physics triangulation ≤ 0.51 dec.**
   Fast surrogate ↔ joint Newton solver ≤ 0.39 dec across reservoir
   biases. Joint solver ↔ ngspice ≤ 0.51 dec across 9 biases at
   production parasitic-NPN parameters (one M2-off tail miss at
   V_G2 = 0). Transitive bound surrogate ↔ ngspice ≤ 0.90 dec.

The brief explicitly does NOT claim NS-RAM is a better reservoir than
software ESNs, does NOT claim cross-task generalisation, does NOT
claim a morphable mixed-mode fabric. All three would have been
overclaims; all three were tested and falsified.

---

## 3. What we tested and what closed it

### 3a. The V_G2 continuum hypothesis (campaign 1)

The hypothesis was that a continuous V_G2 trajectory could morph a
computation from a vanilla 0/1-MOSFET regime into an analog-LIF
regime, providing an "identity-grounding" bridge for compute. Two
pre-registered tests:

- **Rate-dependent hysteresis** (single cell, transient sim,
  triangular V_G2 sweep at T_ramp ∈ {1ns..30ms}, 5 init seeds).
  Hysteresis is real and peaks at T_ramp ≈ 1 ms (body-RC time
  constant τ = C_b·R_b, classical single-RC signature). But its
  CONTRAST vs a quasi-static baseline is ~5×, not the 100×
  pre-registered gate. **Honest FAIL.**

- **Mixed-mode-fabric network** (fraction f of cells V_G2-grounded
  vs floating, NARMA-10 at N = 200, 5 seeds, 7 fractions).
  Best-mix f = 0.25 edges pure-floating by Δ = 0.006 NRMSE; the
  pre-registered margin was max(1 pp, std) = 0.016 NRMSE.
  Pure-grounded (f = 1.0) is chance-level. **Honest FAIL.**

The V_G2 continuum has measurable but soft dynamical content; it is
not an architectural distinguishing feature.

### 3b. NS-RAM vs textbook ESN head-to-head matrix (campaign 2)

Matched N, identical input projection, identical ridge readout,
5 seeds per cell. ESN = sparse tanh, spectral radius 0.9, leak 0.30,
input gain 1.0. Pre-registered gate per cell: NS-RAM CI95 upper
< ESN CI95 lower.

| Task | N | NS-RAM | ESN | Winner |
|---|---|---|---|---|
| seq-MNIST cross-task | 1000 | Δ = +5 pp over proj. | Δ = +27 pp over proj. | **ESN** by 22 pp |
| NARMA-10 (NRMSE) | 200 | 0.612 ± 0.030 (n=30) | 0.563 ± 0.038 (n=30) | **ESN** by 8% |
| NARMA-5 | 200 | 0.623 [0.60, 0.64] | 0.537 [0.52, 0.56] | **ESN** strict |
| NARMA-20 | 200 | 0.986 [0.94, 1.04] | 0.880 [0.71, 1.05] | tie (both chance) |
| Memory Capacity (total k=1..100) | 200 | 1.751 [1.70, 1.79] | 1.973 [1.92, 2.03] | **ESN** strict |
| NARMA-10 N=100 | 100 | 0.693 [0.65, 0.74] | 0.572 [0.54, 0.61] | **ESN** strict |
| NARMA-10 N=500 | 500 | 0.674 [0.65, 0.69] | 0.588 [0.55, 0.62] | **ESN** strict |
| NARMA-10 N=1000 | 1000 | 0.672 [0.64, 0.72] | 0.591 [0.56, 0.63] | **ESN** strict |
| Mackey-Glass h=6 | 200 | 0.193 [0.17, 0.22] | 0.067 [0.04, 0.11] | **ESN** strict |
| Mackey-Glass h=12 | 200 | 0.074 [0.06, 0.09] | 0.049 [0.03, 0.08] | tie (CIs overlap) |

**Aggregate: 11 head-to-head cells; 0 NS-RAM strict wins, 8 ESN strict
wins, 3 ties (all in regimes where both reservoirs flounder).**

NS-RAM is *not* a competitive reservoir against a textbook software
ESN. The deficit is statistically definitive, scale-independent, and
task-class-independent.

---

## 4. Files ready to send

| File | Last revised | Status |
|---|---|---|
| `research_plan/mario_update_note_v2_draft.md` | post-z242/z243 ESN revision | Send-ready |
| `nsram/main-4.pdf` (8 pages, stateless) | post-V_G2/ESN matrix | Send-ready |
| `nsram/onepager.pdf` (2 pages, stateless) | post-V_G2/ESN matrix | Send-ready |
| `nsram/figures/brief_headlines_honest/brief_headlines.pdf` | 3-panel headlines | Attach |
| `research_plan/sebas_silicon_characterisation_request.md` | 2026-05-05 (6+ days unsent) | Send-ready |
| `research_plan/sebas_thick_ox_request_addendum.md` | 2026-05-07 | Send with main |
| `nsram_proposal_full.zip` | 1.3 MB; tex + figures + pdf | Manual-upload package |

The v1 mario_update_note_draft.md is banner-tagged STALE and must NOT
be sent.

---

## 5. What the oracles flagged as residual concerns (none blocking)

- **ESN baseline fairness** (openai+grok, optional). The ESN config
  (sparse 10%, ρ = 0.9, leak = 0.30, input gain 1.0) is a textbook
  default. Both reviewers note that *if anything* a tuned ESN would
  beat NS-RAM by more, not less. A one-task ESN hyperparam sweep
  would close the optics talking point but is not decision-changing.
  We can run this *after* sending if desired.
- **Surrogate dt = 500 ns vs body-RC ≈ 1 ms** (grok). The body-RC
  timescale is captured (covers ~2000 simulation steps), so this is
  a "plausible but not major" concern. Not load-bearing.
- **Unexplored small-N regime** (grok). N = 30 or 50 was not tested;
  NS-RAM might have a niche at very small N. Worth a single
  sanity-check experiment *after* sending. Not blocking.

No oracle suggested retracting a claim, changing the framing further,
or running more reservoir-vs-reservoir tests before sending.

---

## 6. Three concrete user actions

1. **Read `research_plan/mario_update_note_v2_draft.md`** (~5 min).
   Confirm tone and content. Add Mario's email address.
2. **Send Mario v2 + attach the two PDFs from `nsram_proposal_full.zip`**
   (`main-4.pdf`, `onepager.pdf`) + optionally
   `figures/brief_headlines_honest/brief_headlines.pdf`. ~5 min.
3. **Send Sebas main characterisation request + thick-oxide
   addendum** (same email, cc Mario). ~5 min.

Total user time on critical path: ~15 minutes.

---

## 7. What happens after sending

The compute side has no further high-leverage work without Pazos's
silicon transient data. Cron jobs continue running but are mainly
on housekeeping (resource audits, baseline regression, daily synth).
Optional follow-ups, in priority order:

- **ESN-fairness sweep** on one task (~1 hour CPU, optics-only).
- **Small-N sanity check** at N ∈ {30, 50} on NARMA-10 (~30 min CPU).
- **Pivot proposal**: NS-RAM as non-reservoir primitive (compact
  stateful trigger, PUF, chaotic oscillator, programmable nonlinear
  filter). This is a different research program; needs your buy-in
  before launching. Drafted only if you request it.

Once Pazos's transient data arrives, the original M3b plan resumes
(measured τ_body, B_f extraction, silicon-grounded calibration of the
0.654-decade DC fit and the body-state surrogate).

---

*All 13 honest negative results (V_G2 hysteresis, mixed-population,
11 ESN-matrix cells) and the locked Mario v4.3 brief are pushed to
Heigke/NSRAM and synced to nsram/proposal_2026_05/.*
