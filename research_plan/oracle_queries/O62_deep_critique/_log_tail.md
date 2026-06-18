Model rebuild = multi-week, not multi-hour.

Recommend: stop adding fixes for this sprint. Document v4.4 with z304
"baseline" reframed as "spurious local optimum that masks structural
incompleteness", lead with HDC + RNG.

## 2026-05-13 20:13 — R-phase progress check
R-1..R-7 closed (R-4b, R-5, R-6-lite, R-7 all returned). R-8..R-10 NOT
dispatched. Active: O59 oracle critique to distinguish 3 hypotheses for
body-KCL dead path. ALERT: R-phase gate ambiguous — z324 cell-wide
3.25 dec > v5b 3.01 > z304 0.99 → REGRESSED from regression. v4.4 status
unchanged (HDC+RNG locked).

## 2026-05-13 20:30 — O59 3/3 consensus: hypothesis (b) — Iii not wired to Vb

gpt-5 + grok identical diagnosis (gemini Q1 truncated but agrees in synthesis):
"pyport BSIM4 implementation misses IMPACT_IONIZATION wiring to body"

Specific evidence chain:
- iii_kill (alpha0=1e-20) bitwise identical → Iii doesn't reach Vb residual
- Js invariance 11 OoM → diode branch inactive
- Sebas mail: "drop avalanche, BSIM4 + parasiticBJT enough" — REQUIRES Iii→Vb
  coupling for snapback firing

Dispatched R-9: instrument + fix
- Print every Vb-current source at strong bias (V_G1=0.6 V_G2=0.20 V_d=2.0)
- Find alpha0/IIMOD/impact_ionization code path
- Verify it routes to Vb residual (or Sint-only?)
- If missing: add 2-5 LOC fix
- Re-run V_G1=0.6 to confirm body now lives

Gate: V_G1=0.6 log-RMSE changes >5% (any direction) when fix applied.

## 2026-05-13 21:00 — R-9 ROOT CAUSE: Vb pinned to Vd at solver fixed-point

R-9 instrument verdict: Iii IS routed correctly to Vb residual (lines 816,
825, 838, 849 of nsram_cell_2T.py). Hypothesis (b) FALSE.

REAL root cause: solver converges with Vb=2.0V (= Vd) at the test bias.
At this fixed point:
- Vbe = Vb − Vsint = 0.13V → NPN can't fire
- Vds − Vdseff = 0.095V, beta0=20 → T1·exp(−211) = 1e-94 → Iii = 7.5e-48 A
- Body pinned to drain because no body-pull-down path is active

z304's 0.99 dec succeeded because the AVALANCHE PATH (which we removed
per Sebas mail) PULLED Vb DOWN from Vd. That enabled:
- Vbe > 0.5V → NPN fires
- Vds − Vdseff > 0.3V → Iii non-zero → snapback loop closes

v5b regression: we removed avalanche without installing alternative
body-pull-down. Solver finds trivial Vb=Vd fixed-point. All "physics"
silently dies.

**Next experiment**: force Vb-initial-guess away from Vd, OR re-enable
ONE path that pulls Vb down (slightly leaky vnwell→Vb diode at low
saturation current), to bootstrap the solver into the right basin.

This is the deepest insight today: model isn't missing physics — it's
finding the WRONG fixed point because no path is mathematically forcing
Vb < Vd. Sebas's LTSpice probably bootstraps differently via
DC-sweep continuation; our pyport solves cold-start at one bias.

R-9 closes the structural-investigation arc. The remaining question is
NUMERICAL: how to ensure solver finds the physical (Vb < Vd) basin.

## 2026-05-13 20:47 — :47 idle — ACTIVE: z323_v, APU=44C

## 2026-05-13 21:13 — R-phase progress check
R-1..R-9 done. R-9 = ROOT CAUSE: solver finds Vb=Vd trivial fixed-point;
no body-pull-down path active in current pyport. R-10 NOT dispatched
(numerical solver fix: homotopy + weak vnwell→Vb seed leak to bootstrap
solver to physical Vb<Vd basin). User-gated.

## 2026-05-13 21:47 — :47 idle — idle, APU=38C

## 2026-05-13 21:47 — deep-dive 2h cron: 4A-E closed, R-1..R-9 closed, R-10 user-gated, no auto-launch

## 2026-05-13 21:50 — 4h campaign check
P1-P8 master fix: CLOSED. R-1..R-9 topology rebuild: CLOSED with R-9 root cause.
R-10 numerical solver fix: pending user approval. No new ALERT.

## 2026-05-13 22:13 — R-phase progress check
R-1..R-9 closed. R-9 root cause: Vb pinned to Vd at solver fix-point.
R-10 (numerical solver homotopy fix) NOT dispatched, user-gated.
No ALERT.

## 2026-05-13 22:30 — 3h campaign cron: idle, APU=37C, R-10 user-gated, no auto-launch

## 2026-05-13 22:32 — R-10 numerical solver fix DISPATCHED

Pre-reg locked gates:
- INFRA: solver converges to Vb < Vd at V_G1=0.6, V_G2=0.20, V_d=2.0
- PASS: cell-wide median log-RMSE < 0.95 dec (beats z304 0.99)
- AMBITIOUS: < 0.5 dec
- DIAGNOSTIC: V_G1=0.2 < 2.5 dec AND V_G1=0.6 < 0.7 dec (per-branch parity)

## 2026-05-13 22:50 — R-10 ALL strategies FAILED — deeper bug

S1+S2+S3 all give Vb=2.0 (=Vd). Trivial fix-point is MATHEMATICALLY VALID
because residual at Vb=Vd is ~0 (all currents 1e-48). Initial-guess
doesn't help: solver finds the only equation-satisfying point.

Real issue: M1 at V_G1=0.6, V_d=2.0V should generate Iii ~1e-21 A by
BSIM4 IIMOD formula. Our pyport computes 1e-48. Either:
(a) BSIM4 IIMOD formula has implementation bug in pyport
(b) M1's Iii is routed to M1's own body, not the shared/floating Vb
(c) Pyport's body-node connectivity is wrong (M1 body and shared body are
    separated when they should be merged)

Next R-phase = code-level debug of BSIM4 _eval_mosfet → Iii computation
+ trace where M1's Iii goes in residual stamp.

Snapback graphs blocked until M1 Iii physically lives.

NO automatic graphs generated this run — R-10 halted at infra-gate.

## 2026-05-13 22:53 — 6h track audit
Phase A: A.1/A.2 ✓, A.3/A.4 deferred (model-blocked). 2/4.
Phase B: DS-N1✓ DS-N2✓ DS-N3✓(AMBITIOUS+NIST) DS-N5✓(LOCKED 83.86% n=10) DS-N4 in_progress DS-N6✓(FAIL). 5/6.
Phase C: 4A-E✓ brief compiled.
Topology campaign: R-1..R-9 closed (root cause: M1 Iii=1e-48 vs physics 1e-21).
R-10 solver-fix FAILED at infra. R-11 (BSIM4 Iii deep-trace) user-gated.
v4.4 status: HDC+RNG headlines locked, model-rebuild deeper than expected.

## 2026-05-13 23:00 — R-11 BSIM4 IIMOD deep-trace dispatched

Bug hypothesis: pyport's compute_iimpact gives 1e-48 A at M1 strong sat;
physics expects ~1e-21 A (27 OoM off). Subagent will:
- Read leak.py compute_iimpact
- Instrument at V_G1=0.6 V_d=2.0
- Compare to hand-calc step-by-step  
- Identify where divergence happens
- Fix and verify Iii > 1e-25 A
- Re-run V_G1=0.6 to see if cell-wide moves

## 2026-05-13 23:25 — R-11 CLARIFIER: IIMOD code correct, real bug is Vsint pin

R-11 audit: leak.py compute_iimpact matches BSIM4 v4.8.3 §6.1 spec exactly.
Port-vs-hand zero relative error. Iii at Vsint=0 (M1 strong-sat) = 1.18e-11 A
which is PHYSICALLY CORRECT and 14 decades above the 1e-25 INFRA gate.

What z325/z326 reported (1e-48) was at CONVERGED Vsint=1.867V, where
Vgs_M1 = V_G1 - Vsint = 0.6 - 1.867 = -1.27V → M1 OFF → Iii naturally → 0.

**Reframed root cause**: solver finds Vsint=Vd trivial basin where M1
is OFF. R-10 initial-guessed Vb, but it's VSINT that needs to start
at 0 (where M1 conducts, charges body, lifts Vb from 0).

Real solver fix: Vsint_init = 0 (NOT Vb_init = Vd-0.7). Or arclength
homotopy in Vd-Vsint joint.

R-11b oracle 3-way still in flight (code review). Will confirm or 
differ. R-12 = new solver-fix subagent on correct variable.

## 2026-05-13 23:47 — :47 idle — ACTIVE: z326_solver_fix, APU=44C

## 2026-05-13 23:50 — 🎯 R-12b + R-12c CONVERGE: pyport pins Vb=2.0 (wrong)

3 independent tracks confirm same bug:

**R-12c ngspice xcheck** at V_G1=0.6, V_G2=0.20, V_d=2.0:
- ngspice: Vsint = 0.382 V (M1-ON), Vb = 0.267 V (floating)
- pyport: Vsint = 1.867 V (M1-OFF), Vb = 2.0 (PINNED)
- |ΔVsint| = 1.485V. Bug DEFINITIVELY confirmed.

**R-12b empirical Iii(Vsint) map**:
- 25/25 fittable biases have M1-ON basin at low Vsint
- Median Vsint_transition = 0.653V (range 0.57-0.69)
- Heatmap PNG: results/z329_iii_vsint_map/iii_heatmap.png

**Critical finding**: pyport's `_solve_at_fixed_vb` PINS Vb=2.0V. Real
solver (ngspice) lets Vb FLOAT and finds Vb=0.267V. The pin makes
physical solution UNREACHABLE.

R-10's Vb_init=Vd-0.7 didn't help because solver immediately pinned
Vb back to Vd. R-12's Vsint=0 also fails if Vb is still pinned.

**Real R-13 fix**: refactor solver to make Vb a FREE variable, not
pinned. Then Newton + initial guess can find physical basin.

R-11 / R-12 / R-11b still running but will likely converge same diagnosis.

## 2026-05-14 00:05 — 5-track convergence on SAME root cause

R-11 runtime, R-11b static review, R-12c ngspice xcheck, R-12b empirical
Iii map, R-12d snapback graph — ALL converge on:

BUG IS BASIN-OF-ATTRACTION:
- pyport _solve_at_fixed_vb pins Vb, Newton finds Vsint=Vd trivial point
- M1 then OFF (Vgs negative) → Iii vanishes → Vb=Vd self-consistent but
  physically wrong
- ngspice with same cards finds Vsint=0.38, Vb=0.27 (M1-ON basin)
- z331 graph proves model SHAPE correct under forced Vsint

R-13 = 2D Newton (Vsint, Vb both free) with Vsint_init=Vd/2, Vb_init=Vd/2.
Should converge to ngspice-matching basin and give physical amplitude.

z331 snapback graphs SAVED — first time we have model curves showing
knee/dropoff structure. Amplitude 6 OoM off (forced-Vsint floor), R-13
fixes that by letting Vb rise.
