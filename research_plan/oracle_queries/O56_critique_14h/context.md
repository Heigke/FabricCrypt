   - V_peak(slew) = +0.15 V/dec

**Implication for P3 (pyport_v4 build)**: 
- Don't write new trap reservoir — just activate existing TAT card values
- Don't sweep Rs — replace with per-V_G1 R_body
- Add snapback-peak gate as falsification metric

Pitfalls flagged:
- `trise` in CSV (9-13 range) is unitless fit param, NOT literal µs/ms.
  Need Sebas confirm before wiring as ramp-rate covariate.
- OriginLab .opj raw streams unparsed; true trap τ still implicit.
- N-well doping density not in materials (PDK proprietary, use 5e17 cm⁻³).
- V_d > 2.5V quantitative data still gap; >3V tail unconstrained.

**Both P2 gates PASSED**: 9 new signals (≥3 conservative), 4 quantitative
constraints (≥1 ambitious).

## 2026-05-13 11:55 — P1 oracle locked + drift acknowledged

**P3 fix-order locked**: (1) VNwell polarity, (2) rbodymod/Rb-distributed,
(3) drain avalanche, (4) falsifier harness, (5) N1 traps LAST, (6) SRH if needed.

**KWS keep as side experiment** (not headline). HDC+RNG lead.

**Drift acknowledgements** (gpt-5 flagged):
1. z312 N=16384 84.09% headline language tightened: n=4 only, pre-reg was n≥10.
   Restating as "headline candidate, n=10 lock pending". Submitting 6 more seeds.
2. "Noise-benefiting" Δ=0.18pp vs std=0.20pp on n=4 → restated as "no worse 
   than σ=0 within seed noise". 
3. Diode-fix causal claim language stripped until z310b numbers exist.

Submitting 6 more z312 N=16384 seeds now (zgx primary).

## 2026-05-13 11:22 — z313 pyport_v4: FALSIFIED on DC, partial PASS on snapback law

P3 ran. P1 oracle-locked fix combo (polarity + R_body distributed +
drain avalanche, with optional TAT) DEGRADED DC fit:
- Best config (RUN A no TAT): cell-wide 2.91 dec (z304 0.99, Δ=-1.92)
- RUN B (with TAT): V_G1=0.2 med=6.14 / 0.4 med=3.81 / 0.6 med=1.83
- Signed bias hugely positive → model OVER-predicts (+5.98/+4.25/+1.89)
- Slide V_d>2V samples: log-RMSE 1.99 (bad)
- DC verdict: FAIL / FALSIFIED per P1 gate

**Snapback law partial PASS**:
- V_G2=0.05: V_peak_sim=2.741, law=2.699, Δ=+0.042V (within 0.2V gate)
- V_G2=0.10: Δ=-0.117V (within gate)
- V_G2=0.20/0.30/0.50: no Sebas V_G1=0.3 row to interp params → unscored

**Diagnosis**: combining 3 fixes simultaneously created destructive
interactions. Specifically per-V_G1 R_body {1e10, 1e9, 1e8} may be
mis-magnitude — too restrictive at V_G1=0.2 (starves), too leaky at
V_G1=0.6 (dumps).

**Next**: isolate ONE element at a time per scientific bisection:
- z313b: polarity fix ALONE (no R_body, no avalanche)
- z313c: polarity fix + drain avalanche
- z313d: polarity + R_body alone
- z313e: polarity + avalanche + R_body (re-run with refined R_body table)

snapback law-gate found AS WORKING METHOD — keep using.

## 2026-05-13 11:40 — z313 bisection: P1 #1 falsified, infrastructure gap exposed

z313b/c/d/e all 2.905 dec BITWISE IDENTICAL. Means:
- R_body table (cfg.vnwell_Rs) has ZERO effect on _residuals (not wired)
- Drain avalanche (cfg.use_lateral_collector) has ZERO effect (not wired)
- 100% of -1.92 dec regression caused by polarity flip alone

**Root cause** (clean): z304 baseline polarity is correct (reverse-biased
normal operation). z310 had bug WHERE polarity was wrong-signed forward.
z313 "fixed" by flipping again — but z304 was already right side. Two
wrongs DON'T make a right.

**ORACLE P1 RECOMMENDATION #1 WAS WRONG**. Will report back to oracles
for self-correction.

**Per-V_G1 signed bias z313**:
- V_G1=0.2: -1.48 dec
- V_G1=0.4: -3.15 dec  
- V_G1=0.6: -4.61 dec
- Strictly monotonic negative with V_G1 → body-branch coupling issue

**Infrastructure gap**: cfg.vnwell_Rs and cfg.use_lateral_collector flags
are parsed but not consumed by current pyport _residuals. Need code
audit + unit test BEFORE any further parameter sweep makes sense.

**Action plan revised**:
1. Revert z313 → drop polarity flip; restore z304 polarity
2. Audit nsram/bsim4_port/_residuals to find which cfg flags are actually
   live vs orphan; unit-test each before sweep
3. THEN P1 #2 (rbodymod=1 / distributed Rb) — must be CODE work (DBR
   implementation), not flag-flipping per N3 (z309) earlier finding
4. THEN avalanche M(V_bc) — also CODE work, not flag

These are deeper changes than oracle estimated. Multi-day work, not 3h.
Honest: P3 won't close cell-wide < 0.5 dec in current sprint without
real code-side investment.

**Pragmatic v4.4 path** (revised):
- Lead with HDC N=16384 (n=10 lock in flight via z312b queue)
- Lead with Bayesian RNG (NIST 5/5, ESS 1.03×)
- Model section: state z304 0.99 dec as best, snapback gap OPEN, 
  V_d>2V validation set from O52 as new artifact, traps confirmed
  mechanism but unimplemented in production model
- Snapback peak law: USE the 2.73-0.625·V_G2 finding as evidence of
  trajectory, even though pyport doesn't yet reproduce it

## 2026-05-13 12:47 — :47 idle — idle, APU=45C

## 2026-05-13 12:25 — O55 consensus — Option C + drift corrections

3/3 oracles agreed: **Option C** (snapback peak law sweep) is single highest-
value 1-3h experiment.

Q1 most-defensible: HDC N=16384 at n=10. M2 traps NOT shippable (stub
overshoots measured hyst by 14×). T2 = infrastructure not headline.

Q2 cherry-pick CONFIRMED: z304's "0.99 dec baseline" without CI is asymmetric
rigor vs z313's multi-variant. Fix: row-bootstrap 1000 resamples → 95% CI.

**Corrective actions logged**:
- Restate z312 as "headline candidate, n=10 lock pending" (drop "locked" at n=4)
- Drop "noise-benefiting" — say "no worse than σ=0 within CI"
- Restate M2 traps as "qualitative mechanism validation; magnitude not matched"
- Strip diode-fix causal claims pre-z310b
- Bootstrap z304 baseline → real CI

**Launching now**:
1. z317 snapback peak law sweep (Option C) — Sebas data at V_G1 = 0.4 (avoid
   the 'no Sebas row' issue at V_G1=0.3), sweep V_G2 ∈ {0.05, 0.10, 0.15, 
   0.20, 0.30, 0.45}, locate V_peak per curve, compare to 2.73-0.625·V_G2
2. z318 z304 row-bootstrap CI (cheap, parallel)

## 2026-05-13 12:30 — z318 bootstrap CI: revealed bimodal hiding under "0.99 dec"

z304 median 0.988 dec hides bimodal distribution:
- V_G1=0.6 branch: per-row log-RMSE ~0.1 dec (excellent)
- V_G1=0.2 branch: per-row log-RMSE ~4.7 dec (catastrophic)
- Row-bootstrap 95% CI = [0.93, 2.22] reflects the inter-cluster variance

z313_A: median 2.91 dec, 95% CI [2.09, 3.47]
Δ vs z304: +1.917 dec, 95% CI [+0.64, +3.00] — regression CONFIRMED (CI excludes 0)
   But: magnitude uncertain by ~2×

**Sharpened language for v4.4**:
- DO NOT say "z304 0.99 dec baseline" — it's bimodal
- DO say: "median 0.99 dec masking strong V_G1-stratification 
  (best branch 0.1 dec, worst 4.7 dec)"
- DO say: "z313 regression statistically confirmed but magnitude has
  ~±2× CI"

Both gates passed: CI reported, Δ retains significance.

Pitfalls flagged:
- Asymmetric row counts (z304=33 vs z313=25) — paired analysis needs refit
- Bimodal small-N median bootstrap → bias; BCa preferred over percentile

## 2026-05-13 12:35 — z317 snapback peak law: PASS conservative, BONUS slope FAIL

V_G1=0.4 sweep, 6 V_G2 points. 4/6 within 0.3V after V_G1-shift of +0.752V.

Results table:
| V_G2 | V_peak_sim | V_peak_law | Δ_shifted |
|---|---|---|---|
| 0.05 | 1.200 | 2.699 | -0.747 (OUTLIER, pulls slope) |
| 0.10 | 1.900 | 2.667 | -0.016 |
| 0.15 | 1.900 | 2.636 | +0.016 |
| 0.20 | 2.250 | 2.605 | +0.397 |
| 0.30 | 1.700 | 2.542 | -0.091 |
| 0.45 | 1.800 | 2.449 | +0.103 |

Linear fit slope: **+0.611** (law: −0.625 — SIGN FLIPPED). V_G2=0.05 outlier
drives this; removing it leaves cluster at 1.7-2.25 with no monotonic trend.

Gates:
- PASS-conservative: PASS (4/6 ≥ 4 threshold)
- AMBITIOUS (5/6 within 0.2V): FAIL
- BONUS slope: FAIL (sign-wrong)

Honest verdict: snapback peak EXISTS in pyport at V_G2 ≥ 0.10 (cluster
within ±0.4V of law) but the V_G2-slope is wrong-signed (model thinks
higher V_G2 → higher V_peak; law says opposite). This is real physics
diagnostic: the model's mechanism for V_peak modulation by V_G2 has
incorrect polarity. NOT the same bug as the polarity flip from z313 —
this is about HOW V_G2 modulates the snapback location.

V_G2=0.45 row uses interpolated closest Sebas row (max 0.30) — listed
as caveat.

Rate-probe was crude (V_d step size, not actual slew rate). To test
+0.15 V/dec slew law properly: need controlled-t_rise transient, not DC.

**v4.4 brief language**: "Snapback peak qualitatively reproduced at
V_G2 ≥ 0.10; quantitative V_G2-slope inverted, indicating mechanism
direction error that the topology rebuild must address."

Both z317 + z318 close out O55 Option C and Q2 cherry-pick fix.

## 2026-05-13 12:38 — 🎯 v4.4 HEADLINE LOCK (n=10 confirmed)

z312b 12 seeds + z312 4 seeds aggregated:

| condition | n=10 mean | std | CI95 width |
|---|---|---|---|
| N=16384 σ=0.00 | **83.86%** | 0.28pp | 0.34pp |
| N=16384 σ=0.05 | **83.87%** | 0.40pp | 0.49pp |

**Locked v4.4 headline**: HDC N=16384 → 83.86% UCI-HAR at n=10, 
35 nJ/inference, CI95 ±0.17pp.

**Drift correction applied**: σ=0 and σ=0.05 means are 83.86 vs 83.87 with
fully-overlapping CI. Earlier "noise-benefiting" claim REJECTED at n=10.
Restated: "noise-tolerant" (no degradation in CI, but no benefit either).

Per cron protocol: AMBITIOUS PASS at n≥10 → v4.4-BREAKTHROUGH locked.

## 2026-05-13 13:47 — :47 idle — idle, APU=41C

## 2026-05-13 13:47 — Deep-dive 2h cron: triggering 4E brief compile
APU=41°C. 4A-D closed. n=10 headline lock confirmed at 83.86% (Δ=0.17pp CI).
Dispatching 4E compile subagent.

## 2026-05-13 13:50 — Campaign 4h progress check
P-phase status: P1✓ P2✓ P3 FALSIFIED P4 deprioritized P5/P6 blocked P7 partial (O55) P8 IN FLIGHT.
v4.4 headline LOCKED n=10 at 13:38. 4E compile subagent active (a6f9615).
No ALERT: P8 is the active phase, no gate crossing pending dispatch.

## 2026-05-13 14:00 — 4E v4.4 brief compiled (P8 done)

`research_plan/4E_v4.4_brief.md` — 2654 words, 10 sections, all-cards-on-table.

Locked headlines verified:
- HDC 83.86% n=10 CI95 ±0.17pp (z312/z312b)
- Bayesian RNG NIST 5/5 (z296b)
- Snapback peak law 4/6 within 0.3V (z317)

Honest gaps stated: bimodal z304, slope sign-inverted, infrastructure inert.

Still-missing pre-send-to-Mario:
- Explicit Appendix A "ask" block (V_d>2V multi-rate transient + wait-time exp)
- Figures (HDC N-scaling, per-row log-RMSE heatmap, snapback law fit)
- 35 nJ/inf derivation footnote

Phase status: P1-P3 done, P4 deprioritized (KWS at chance), P5/P6 blocked
on infrastructure, P7 partial, P8 DONE. Master fix plan complete.

Next user gates:
1. Generate figures + appendix → send Mario
2. OR: prioritize topology rebuild (multi-day code work)
3. OR: keep accumulating side-results
