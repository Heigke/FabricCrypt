on infrastructure, P7 partial, P8 DONE. Master fix plan complete.

Next user gates:
1. Generate figures + appendix → send Mario
2. OR: prioritize topology rebuild (multi-day code work)
3. OR: keep accumulating side-results

## 2026-05-13 13:25 — O56 critique synthesis

Q1: HDC LOCKED data-stark, but "shippable" framing fragile due to inert
cfg flags (tech debt). HDC uses V_G1=0.3 (interpolated), and bimodal V_G1=0.2
catastrophe could leak into network performance.

Q2 (sharp): Rerun HDC at V_G1=0.2 (catastrophic DC branch). Pre-reg null
hypothesis: accuracy ≥ 80% there. If acc < 80% → headline robusthet
FALSIFIED (DC bias DOES leak into network).

Q3 NO-CHEAT: 3/3 say discipline HOLDS since O55. All drift corrections
applied. No new flag.

## 2026-05-13 13:25 — Pre-reg z319: V_G1=0.2 HDC falsifier (Q2 from O56)

LOCKED gate:
- PASS (claim defended): mean test_acc ≥ 80% at V_G1=0.2, σ=0, N=16384, n=4
- FAIL (claim falsified): mean < 80% → HDC headline robusthet has hidden 
  V_G1-dependence we haven't acknowledged
- AMBITIOUS: mean ≥ 83% (within 1pp of V_G1=0.3 headline) → strong defense

Cluster idle, APU 41°C. Submitting now.

## 2026-05-13 14:47 — :47 idle — idle, APU=43C

## 2026-05-13 15:47 — :47 idle — idle, APU=37C

## 2026-05-13 14:51 — 🎯 z319 AMBITIOUS PASS — HDC ROBUST to DC bimodal

V_G1=0.2 (catastrophic DC branch 4.7 dec off) at N=16384 n=4:
- seed 0: 83.75%
- seed 1: 83.64%
- seed 2: 84.02%
- seed 3: 84.22%
- mean = 83.91% (V_G1=0.3 headline: 83.86% → within 0.05pp)

**Profound finding**: HDC encoding is INSENSITIVE to per-cell DC fit
quality. The "model 10× off at V_G1=0.2" does NOT leak into network
performance. HDC's noise-averaging-over-N=16384-cells dominates per-
cell signal anyway.

**Gate verdict**:
- PASS (≥80% defense): PASS (83.91% ≫ 80%)
- AMBITIOUS (within 1pp of V_G1=0.3): PASS (within 0.05pp — far better than 1pp)

**O56 Q2 falsification DEFEATED**: gemini hypothesis "if V_G1=0.2 acc 
drops below 80% → headline robusthet falsified" — REJECTED. V_G1=0.2 
yields essentially identical accuracy as V_G1=0.3.

**v4.4 brief update needed**: §6 honest gap "z304 DC bimodal at V_G1=0.2"
remains valid for DC fit, but should add §4.1: "Network-level performance
is empirically robust across V_G1 branches — z319 V_G1=0.2 yields
83.91% (within 0.05pp of V_G1=0.3 headline) despite DC catastrophic at
that bias point. HDC's noise-averaging dominates per-cell DC errors."

This is a v4.4 STRENGTH not a weakness.

## 2026-05-13 14:55 — 3h campaign cron: PLAN CLOSED + z319 defense already logged

State: 4E brief compiled, n=10 headline locked (83.86%), z319 defended
robustness against bimodal DC, O56 critique → no new drift flags.
Cluster idle 37°C. No new compute launched this cron — would duplicate.

This phase of campaign closed. Remaining items for v4.4 ship:
- Generate figures (HDC N-scaling curve, snapback law fit, per-row heatmap)
- Add Appendix A "ask" block for Sebas
- 35 nJ/inf derivation footnote
- All user-gated, not auto-cron-launched.

## 2026-05-13 14:55 — 6h track audit
Phase A: A.1/A.2✓ A.3/A.4 deferred. 2/4.
Phase B: DS-N1✓ DS-N2✓ DS-N3✓(AMBITIOUS+NIST) DS-N5✓(LOCKED v4.4 headline) DS-N4 in_progress(#186) DS-N6✓(FAIL). 5/6.
Phase C: 4A-D✓ 4E✓ compiled.
Today's biggest wins: z319 defense (HDC robust to DC bimodal) + 4E brief compiled + Swedish podcast 22min.
v4.4 status: HEADLINE LOCKED+DEFENDED, brief drafted, pre-send items = figures + Appendix A user-gated.

## 2026-05-13 15:00 — TOPOLOGY REBUILD CAMPAIGN R-1...R-10

Plan: research_plan/TOPOLOGY_REBUILD_PLAN_2026-05-13.md
Cron: 0e1bec5c (hourly :13 progress check)

MASSIVE find: nsram/Zoom/ dir NEVER audited — has Zoom meeting transcript
with Sebastian + slow-IV sweeps (SRavg=0 = DC limit, NEW DATA) + BSIMfitsBA.

R-1 + R-3 audit subagents launched in parallel (research only, no compute).
After they land → R-4 pyport_v5 build (real _residuals wiring).
Stop criterion: snapback + DC + transient + LIF all in approximate right
range, model not missing physics. Then network sim.

## 2026-05-13 16:47 — :47 idle — idle, APU=39C

## 2026-05-13 16:55 — 🎯 R-1 = MASSIVE UNLOCK (8 new signals from mail.txt)

User correctly flagged transcript as "jättedåligt" — but the parallel
mail.txt + raw M1/M2 cards yielded 8 new signals:

**5 game-changing**:
1. Cb = 5-10 fF (vs our 1 fF, 10× TOO SMALL — body τ scales 7×)
2. pdiode area = 22 μm² (5×4.4) (vs our 1u placeholder, 22× off)
3. LDE stress block on M1 only: saref/sbref=1.04μm, ku0=-2.7e-8, kvth0=9.8e-9
4. parasiticBJT is NOT a real device per Sebas — "complementary firing
   current source", model artifact only. Stop calibrating against real BJT physics.
5. 24 Slow-IV SRavg=0 CSVs (DC-limit data we never used)

**Implication for R-4 v5 build** (in flight):
- Set Cb default 7 fF (not 1 fF). May close transient/hysteresis gap directly.
- Set Adiode = 22 μm² as default. Likely fixes V_G1=0.2 sub-threshold catastrophe.
- Add LDE stress block on M1 → explains etab asymmetry physically.
- Reframe NPN: stop trying to match real BJT, treat as firing-current source.
- Use 24 SRavg=0 CSVs as canonical regression target (DC-limit data, less ramp confound).

Need to inform R-4 subagent OR have new R-4b after this lands.

AMBITIOUS PASS per gate spec. PushNotification triggered.

## 2026-05-13 17:00 — R-1b deep audit: complete v5 recipe + paradigm shift

R-1b processed 31 Zoom images via gpt-5 vision + xlsx + pptx + mail.txt.

**CRITICAL paradigm shift** (mail 2026-04-17): Sebas DROPPED avalanche-
diode path due to LTSpice convergence. Uses BSIM4 IMPACT_IONIZATION +
body-bias dynamics directly. Our pyport has been fighting wrong physics
for snapback.

**Complete v5 recipe** (consolidating R-1, R-1b, R-3):
1. Cb = 7 fF (was 1 fF)
2. Adiode = 22 μm² (was 1μ placeholder)
3. ALPHA0 = 7.842e-5 CONSTANT (drop polynomial)
4. K1 = 3-node V_G1-only LUT (drop V_G2 dependence)
5. mbjt = binary step at V_G1 ≈ 0.3 (off ↔ on)
6. **DROP avalanche / Chynoweth entirely** (per Sebas)
7. Use BSIM4 IMPACT_IONIZATION (alpha0/beta0 already there) + RC body cap
8. LDE stress block on M1 only: saref=sbref=1.04μm, ku0=-2.7e-8, kvth0=9.8e-9
9. parasiticBJT = firing-current-source artifact, NOT real BJT
10. body_pdiode_Rs + harmonic-mean limiter (R-3 wiring fix)

**Brian2 SNN benchmarks**: Poisson=85%, LIF=72% (slide content)
**Energy quoted**: 21.5 pJ/spike (6.7 generation + 25 fJ integration)
**Cell area**: 17 µm² 2T thick-ox, 5.3×6 µm² 130nm

**Slow-IV CSVs**: md5-identical to existing. NO new IV data.
**Fits-xlsx**: only 23 of 33 rows fitted (10 NaN failures).

Now R-4 has full recipe to use. If R-4 lands with limited recipe, launch
R-4b with this complete list.

## 2026-05-13 17:13 — Topology rebuild progress check
R-1 (Zoom audit transcript): DONE, mail.txt 5 game-changers extracted.
R-1b (deep images+xlsx+pptx): DONE, paradigm shift on snapback (Sebas dropped avalanche).
R-3 (pyport infra audit): DONE, ~90 LOC scoping, 8 ORPHAN flags identified.
R-4 (pyport_v5 build): IN FLIGHT (started ~16:00, 70 min in).
R-5 through R-10: BLOCKED on R-4. No new ALERT.

## 2026-05-13 17:30 — R-4 v5 done: INFRASTRUCTURE WIN, params still wrong

z320_pyport_v5 results:
- Unit tests 5/5 PASS (body_pdiode_Rs, enable_tat, tat_vtss, tat_xtss, avalanche)
- Cell-wide median 2.91-3.62 dec, all configs WORSE than z304 0.99
- Verdict: infrastructure now solid, parameters need full R-1+R-1b recipe

R-4 was dispatched BEFORE R-1/R-1b landed. Used old params:
- Cb=1fF (R-1: should be 7fF)
- Adiode placeholder (R-1: 22μm²)
- ALPHA0 polynomial (R-1b: const 7.842e-5)
- Kept avalanche path (Sebas DROPPED per mail 04-17)
- No K1 LUT, no mbjt step

**R-4b launching** with FULL recipe.

## 2026-05-13 17:47 — :47 idle — ACTIVE: z321_pyport_v, APU=44C

## 2026-05-13 17:47 — deep-dive 2h cron: z321 R-4b active, no new launch

## 2026-05-13 17:50 — 4h campaign progress
P1-P8 (master fix): all closed last cycle. New campaign: R-1..R-10 topology rebuild active.
R-1✓ R-1b✓ R-3✓ R-4✓(infrastructure WIN, params FAIL). R-4b IN FLIGHT (full recipe).
R-5...R-10 BLOCKED on R-4b. No new ALERT.

## 2026-05-13 18:13 — R-phase progress check
R-1✓ R-1b✓ R-3✓ R-4✓ infrastructure (5/5 unit tests).
R-4b IN FLIGHT (full recipe + Bf×Js ablation, cell 2/25 ~3.01 dec, ~80min ETA).
R-5...R-10 BLOCKED. No ALERT.

## 2026-05-13 18:47 — :47 idle — ACTIVE: z321_pyport_v, APU=44C

## 2026-05-13 19:13 — R-phase progress check
R-4b z321 IN FLIGHT (cell 2/25 ~3.01 dec mid-run, ~50 min ETA remaining).
No gates crossed since last check. No new dispatch.

## 2026-05-13 18:15 — Pre-reg R-5 (LIF zgx) + R-6-lite (daedalus brute-force)
Locked gates:
R-5: 2T-cell driven by V_G1 pulse → M2 produces ≥1 spike with 0.5-1.5V swing, refractory 1-100 µs. PASS if any (Bf, Js, Cb) config gives clean spike + refractory.
R-6-lite: extend R-4b's Bf×Js grid with K1_LUT_scale × mbjt_step_position sweep on daedalus CPU. Gate: cell-wide median < 1.0 dec on at least one config.
