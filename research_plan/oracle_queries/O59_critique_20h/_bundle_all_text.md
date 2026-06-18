# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (13203 chars) ===
```
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

## 2026-05-13 18:30 — 3h campaign cron: ACTIVE: z323_v, APU=43C (R-4b + R-5 + R-6-lite + 2 structural audits parallel)

## 2026-05-13 18:42 — R-deep-A topology audit: STRUCTURAL BUGS FOUND

3 HIGH-likelihood discrepancies between LTSpice .asc and pyport _residuals:

D1 (root cause): Q1 emitter is GND in pyport, but LTSpice .asc routes it to
Sint rail. "A.1.i finding" from history was apparently wrong — wire trace 
shows E→Sint. NPN sources Ie into Sint in LTSpice; pyport ignores it.

D2: pyport ADDS use_well_diode + body_pdiode_to=vnwell in PARALLEL on same
physical junction. LTSpice has ZERO explicit diodes — Nwell junction is
IMPLICIT inside BSIM4 PTM130bulkNSRAM. We're triple-counting.

D9: parasiticBJT.txt has Bf=10000. Our Bf=50/100/3000 are all wrong vs card.
Sebas mail-Bf=50 was for a different sub-test.

**Fix order**: D1 → D2 → D9. Then re-run V_G1=0.6 — expect recovery toward
z304's 0.43 dec.

This explains the v5b regression (V_G1=0.6 0.43→1.18). Triple-counted
diodes + wrong Q1 emitter wiring + wrong Bf = added "physics" that's
actually wrong-direction.

R-deep-B oracle 3-way still in flight. Awaiting verdict before dispatching 
structural-fix subagent.

## 2026-05-13 18:48 — R-deep-B oracle synthesis + R-7 structural fix dispatch

R-deep-B 3-way (gpt-5+gemini+grok):
- 2/3 structural, 1/3 spurious z304. ALL 3: not pure parametric.
- Body p-n diode DC path INACTIVE (Js bitwise identical = direct proof)
- V_G1=0.6 regression = REMOVAL OF COMPENSATING ERRORS. z304 0.43 leaned on
  non-physical Bf=3000 + avalanche crutch + K1(V_G2) DOF. v5b removed
  crutches but didn't install correct body branch.

Recommended order: make body branch LIVE → verify Js sensitivity returns → BBO.

R-7 z324 dispatched: D1 (Q1.E=GND→Sint, override A.1.i) + D2 (disable
parallel diodes, rely on BSIM internal) + D9 (Bf 10000 from card). Then
Stage 1 liveness ablation (4 variants at V_G1=0.6) + Stage 2 conditional
BBO. Expected runtime 3h.

## 2026-05-13 19:47 — :47 idle — ACTIVE: z323_v, APU=45C

## 2026-05-13 19:47 — deep-dive 2h cron: R-4b + R-5 + R-6-lite + R-7 all active, no new dispatch

## 2026-05-13 19:55 — R-7 z324 STRUCTURAL FIX: STILL DEAD body branch

D1 (Q1.E=GND→Sint), D2 (disable extra diodes), D9 (Bf=10000) all applied.
V_G1=0.6 liveness ablation: V1=V2=V3=3.248 dec BITWISE identical.

Deeper structural issue exposed: pyport body-KCL has NO ACTIVE CURRENT
SOURCE driving Vb away from 0. Without injection, all secondary paths
cancel at Vb≈Vsint≈0 fixed point.

z304's 0.99 dec was POSSIBLE because it relied on (per R-deep-B):
- Bf=3000 NPN compensating for missing path
- Avalanche/Chynoweth crutch
- K1(V_G2) polynomial DOF
Each removed → exposes body-KCL has no real charge-injection mechanism
in our implementation.

The TRUE missing physics may be:
- BSIM4 IMPACT_IONIZATION wiring INTO Vb (not just into Sint)
- GIDL contribution at drain (alpha0/beta0 → currents to body)
- Explicit Iii_to_body coupling not just NPN base current

z304's "fit" was a parameter-coincidence not a physical model.

**HONEST status**: Model is structurally incomplete. Today's R-1..R-7
sequence diagnosed the depth of the problem accurately. Real fix
requires implementing body-injection physics (not just rewiring
existing).

This is the bottom of the honesty stack. v4.4 brief stays correct: 
HDC + RNG headlines hold (network-level robust to model bias per z319).
Model rebuild = multi-week, not multi-hour.

Recommend: stop adding fixes for this sprint. Document v4.4 with z304
"baseline" reframed as "spurious local optimum that masks structural
incompleteness", lead with HDC + RNG.

```


=== FILE: research_plan_R_deep_A_topology_compare.md (8319 chars) ===
```
# R_deep_A — Topology Comparison: LTSpice `2tnsram_simple.asc` vs pyport_v5 `_residuals`

Date: 2026-05-13
Source files:
- LTSpice: `data/sebas_2026_04_22/2tnsram_simple.asc` + `parasiticBJT.txt` + `PTM130bulkNSRAM.txt`
- pyport: `nsram/nsram/bsim4_port/nsram_cell_2T.py` (post R-3+R-4 wiring)

---

## 1. Nodes

| LTSpice .asc        | pyport `_residuals` arg | Notes                                 |
|---------------------|-------------------------|---------------------------------------|
| `Din` / `D`         | `Vd`                    | external pin (input)                  |
| `S`                 | hard-coded `0` (GND)    | external output pin = ground          |
| `Sint`              | `Vsint` (solved)        | floating internal node                |
| `B`                 | `Vb` (solved)           | floating bulk                         |
| `G`                 | `VG1`                   | M1 gate                               |
| `G2`                | `VG2`                   | M2 gate                               |
| (no `Nwell` flag)   | `cfg.vnwell` (param)    | LTSpice does NOT show a Nwell node    |
| GND (flag 0)        | 0                       |                                       |

**LTSpice node count = 6** (D, Sint, B, G, G2, GND). **No Nwell node.**
**pyport node count = 6 solved + Nwell-as-parameter** (one extra virtual node).

## 2. Devices

| LTSpice (4 devices)                                                | pyport (5+ devices)                                                                            |
|---|---|
| **M1** `nmos4` PTM130bulkNSRAM, L=Ln, W=Wn, D=D, G=G, S=Sint, B=B  | M1 BSIM4 D=Vd G=VG1 S=Vsint B=Vb ✓                                                              |
| **M2** `nmos4` PTM130bulkNSRAM, L=10·Ln, W=Wn — D=Sint, G=G2, S=GND, **B = (left unconnected → GND)** | M2 BSIM4 D=Vsint G=VG2 S=0 B=`zero if cfg.m2_body_gnd else Vb` ✓ |
| **Q1** `parasiticBJT` NPN, **C=D, B=B, E=Sint** (per wire trace: pin (752,112) on D-rail; pin (752,208) on Sint-rail; pin (~736,160) on B-rail) | NPN compute_bjt with **Vbe=Vb (E=GND), Vbc=Vb−Vd**  — **EMITTER = GND, NOT Sint** |
| **C1** cap `CBpar` = 1 fF Rser=1m, from **B → GND** (top pin 704,288 ↔ B-net y=160; bottom pin 704,352 → GND flag at 704,416) | NOT in DC residuals (caps inactive in `.op`); `Cbody` only used in transient |
| (none — no pdiode/well-diode device in netlist)                    | **vnwell well-diode** (`use_well_diode`) + **body_pdiode** (`body_pdiode_to`) with optional series-Rs (R-4) + optional TAT current — none of which exists in LTSpice |
| (none)                                                             | iii_gain bookkeeping, lateral collector (Ic_lat = Bf·Ib_lat), avalanche multiplier, local-base inner Newton |

**LTSpice device count = 4** (M1, M2, Q1, C1).
**pyport effective device count = 4 + (1 vnwell + 1 body_pdiode + 1 TAT) = up to 7** in DC.

## 3. Numbered Discrepancies (ordered by likelihood of causing v5b regression)

### D1. **Q1 emitter wired to GND, not Sint**  [HIGH]
- LTSpice wire trace: 800-col gap between y=112 (D-rail) and y=208 (Sint-rail) is exactly where Q1 sits at (736,112). NPN R0 pins land C@(752,112)→D-net, E@(752,208)→**Sint-net**, B@(~736,160)→B-net.
- pyport (line 510-515): `Vbe = Vb` with comment "emitter = ground (legacy F1.v2 path)" — explicit deviation justified by "A.1.i finding" claim.
- Consequence: With E=GND, Q1 turns on at Vb~0.6 V drawing current from D→GND, completely bypassing Sint. With E=Sint (true LTSpice), Vbe = Vb − Vsint, Q1 only fires when Vb leads Vsint — fundamentally different snapback dynamics. **R_Sint also missing a +Ie_Q1 term** (line 535: no BJT current touches Sint at all, but in LTSpice Q1 sources Ie INTO Sint).

### D2. **Extra Nwell-coupled diodes that do not exist in netlist**  [HIGH]
- LTSpice has **zero** explicit diode devices and **no Nwell node**. The N-well/p-substrate junction is implicit in BSIM4 `dnwell` parameters of PTM130bulkNSRAM (handled inside the MOSFET model itself).
- pyport: `use_well_diode=True` (default) injects `I_well_body = mbjt · Js·A·(exp(...)−1)` between a phantom `vnwell` parameter and Vb. Plus `body_pdiode_to="vnwell"` adds *another* parallel diode at the same junction. With Bf=50 and v5b R-4 series-R the body is now pinned to vnwell, killing snapback.
- This explains why "adding physical elements made it worse": LTSpice models the well junction implicitly once, pyport models it explicitly twice (well_diode + body_pdiode) AND adds the BSIM4 internal one. Triple-count.

### D3. **CBpar (1 fF B→GND) missing in DC residuals (silent in `.op`, but flagged because v5b enables transient elsewhere)**  [MED]
- LTSpice C1: B → GND, 1 fF. Inactive in `.op 0` so does not affect DC.
- pyport: `Cbody` parameter exists but is not referenced in `_residuals` at all. **Polarity check**: any transient path elsewhere must use B→GND, not B→Sint.

### D4. **mbjt scaling has no physical analog**  [MED]
- pyport multiplies `I_well_body *= cfg.vnwell_mbjt`. There is no per-bias scaling factor for the well diode in LTSpice (the MOSFET's BSIM4 internal junction is sized by area only).
- This was a fitting kludge to fight D2's overcounting and breaks when v5b switches to Sebas's published Bf.

### D5. **`m2_body_gnd` defaults / branch divergence**  [MED]
- LTSpice: M2.B is **floating-unconnected** in the symbol → LTSpice defaults to `0` (GND) — pyport gets this right when `m2_body_gnd=True`.
- But the residual has two large code branches (`m2_body_gnd` vs not). The "not" branch subtracts `m2["Ibs"]+m2["Ibd"]` from Vb (treating M2.B=Vb) which contradicts the LTSpice schematic.
- Confirm default is `m2_body_gnd=True`; if any v5 caller passes False, body is double-leaked.

### D6. **Series-R on body_pdiode = 1e10 Ω (R-4 default)**  [LOW]
- Without any physical analog. Effectively makes body_pdiode behave as resistor (since exp current dwarfs 1e10 Ω drop only at very high V). LTSpice has no such resistor.
- LOW likelihood as primary culprit (large Rs ≈ disabling it), but interacts with D2 unpredictably.

### D7. **Avalanche multiplier removed per R-1b** [LOW]
- LTSpice PTM130bulkNSRAM uses BSIM4 Iii (impact-ionization) for avalanche. pyport `use_lateral_collector=False` default. Consistent with R-1b mail. No discrepancy in current default.

### D8. **iii_gain inflation in body KCL** [LOW]
- pyport inflates Iii by `iii_gain` (default >1 with sigmoid). LTSpice uses raw BSIM4 Iii once. This is a model-tuning, not topology, divergence.

### D9. **NPN Bf**: parasiticBJT.txt has **Bf=10000** [HIGH context, not strictly a residuals bug]
- The instruction text says "Sebas's published Bf=50" but the model card file shows `bf=10000`. If v5b is using Bf=50 vs the file's 10000, the BJT is 200× weaker — but with E=GND (D1 wrong), even Bf=10000 produces the wrong qualitative behavior.

## 4. Top 3 Fixes (in order)

1. **Fix D1: Wire Q1.E to Sint.**
   - `nsram_cell_2T.py:514-519` — change `Vbe = Vb` → `Vbe = Vb - Vsint` and `Vbc = Vb - Vd` stays.
   - `nsram_cell_2T.py:535` (`R_Sint`) — add **`+ Ie_Q1`** (emitter current into Sint; sign: `Ie_Q1` from `compute_bjt` is current leaving emitter, so flows INTO Sint when BJT is forward).  Verify sign with bjt.py.
   - Expected effect: snapback regime changes from "Vb-only trigger" to "Vb-leads-Vsint trigger" matching LTSpice physics.

2. **Fix D2: Disable extraneous well/body diodes by default.**
   - `nsram_cell_2T.py:117` set `use_well_diode: bool = False`.
   - `nsram_cell_2T.py:162` set `body_pdiode_to: str = "off"`.
   - Rationale: LTSpice models the N-well junction implicitly inside PTM130bulkNSRAM's BSIM4 (`dnwell`/source-bulk diode). Explicit diodes triple-count.
   - If a "vnwell knob" is required for the V_Nwell sweep experiments, expose it ONLY through BSIM4 `nstype`/`dnwell` model parameters, not as an additional diode device.

3. **Fix D9 (sanity): Use parasiticBJT.txt Bf=10000.**
   - Wherever `GummelPoonNPN` is constructed for the 2T cell, source `Bf` from `data/sebas_2026_04_22/parasiticBJT.txt` (`bf=10000`), not from a separate "published" value of 50/100.
   - Combined with Fix 1, will restore the strong reverse-Early/snapback that LTSpice produces.

## 5. Gate Status

≥3 structural discrepancies identified at HIGH likelihood (D1, D2, D9). Gate **OPEN**.

```


=== FILE: research_plan_R_deep_B_oracle_structural.md (5563 chars) ===
```
# R_deep_B — Oracle Synthesis: Structural vs Parametric? (O58_structural)

**Date**: 2026-05-13
**Packet**: `research_plan/oracle_queries/O58_structural/`
**Providers**: openai (gpt-5, 135s), gemini (2.5-pro, 68s), grok (4-latest, 59s)
**Wall**: ~4.4 min

---

## Per-question consensus / dissent

### Q1 — Structural vs parametric vs spurious?

| Oracle | Primary | Secondary |
|---|---|---|
| **gpt-5** | **C (spurious)** | B > A |
| **gemini** | **A (structural)** | C strong secondary |
| **grok** | **A (structural)** | C possible, B unlikely |

**Consensus**: it is **NOT pure (B) parametric**. All three reject "just sweep harder." 2/3 vote structural (A); gpt-5 inverts and says z304 was a spurious local optimum on wrong physics — but it agrees the v5b body-diode path is dead in KCL (which is itself a structural fact). So **the unanimous operational call is: there is at least one dead current path in v5b**, regardless of whether you call that "structural" or "parameter region with a dead branch".

**Dissent**: gpt-5 weights C > A because z304's Bf=9000 best-branch is nonphysical and v5b passes unit tests (no sign catastrophes). Gemini+grok weight A higher because the audit (`R3_pyport_audit.md`) explicitly flagged missing `body_pdiode_Rs` and the Js-invariance is bit-exact.

**Combined verdict**: Both are simultaneously true. z304 was a spurious optimum (fit-by-overfitting via nonphysical Bf and avalanche crutch) AND v5b has at least one structurally inert path (body diode). Removing the z304 crutches before fixing the v5b dead branch produced the regression.

### Q2 — Js invariance → which path dominates?

**Unanimous**: body p-n diode DC path is **inactive / negligible**. The dominant current paths are (in agreement):
1. Channel `Ids` (BSIM4)
2. Impact ionization `I_iii` (BSIM4 ALPHA0/BETA0 → body)
3. Parasitic BJT `I_bjt` (complementary firing source)

Diode role per Sebas: capacitive (Cb, transient time-constant), **not DC firing**. Adiode=22μm² and Cb=7fF make sense only in transient context.

Mechanism for inactivity (gemini + grok converge): missing `body_pdiode_Rs` series resistance means the diode branch is either clamped by the network or never reaches forward conduction in (Vd ∈ [0,2], V_G1 ∈ [0.2,0.6]) — body voltage floats low, diode stays off.

### Q3 — Why did "adding correct physics" regress V_G1=0.6?

**Unanimous mechanism**: *removing compensating errors*.

- z304 had three "crutches" giving it surplus DOF: (i) K1(VG2) instead of K1(VG1), (ii) ALPHA0 polynomial in (VG1,VG2), (iii) active avalanche/Chynoweth path, (iv) nonphysical Bf ≫ 50.
- v5b correctly removes (i)(ii)(iii) per Sebas's recipe and reframes BJT with Bf≈50.
- But v5b did NOT yet rewire the body voltage correctly (diode path dead). So the model lost its crutches before its real replacement mechanism became active → regression, especially at V_G1=0.6 where the avalanche crutch had been doing the most work.

### Q4 — Cheapest 2h discriminating experiment?

All three propose **path-liveness ablation**, differing in implementation:

| Oracle | Design |
|---|---|
| gpt-5 | Toggle 3 mechanisms (`iii_to_body_factor=0`, `mbjt=0`, `body_pdiode_to="off"`) at 3 bias corners. Gate: if A1 and A2 both move <0.1 dec → structural fault. |
| gemini | Single-cell `use_well_diode=True` with vnwell_Rs=1e8Ω vs current control. Gate: >1% rmse change → structural confirmed. |
| grok | 5×5 Bf×Js sweep on full v5b + add body_pdiode_Rs. Gate: any combo <1.0 dec → parametric; all ≥3.0 → structural. |

---

## Verdict

**Structural with parametric-amplification**. The v5b model has at least one dead KCL branch (body p-n diode, Js-invariant by bit-exact test). The previous z304 "success" was a spurious local optimum riding on now-removed crutches (overfit Bf, K1(VG2), avalanche). You cannot resolve this with BBO until the body-voltage path is electrically live.

**Order of operations**:
1. Make the body branch live (add `body_pdiode_Rs` OR re-enable the existing `vnwell_Rs` path).
2. Verify Js sweep now produces non-identical residuals (positive control on structural fix).
3. Then BBO over (Bf, K1_LUT_scale, mbjt_step_threshold, BETA0_scale).

## Recommended cheapest 2h experiment (synthesized)

**Two-stage liveness ablation, ≤2h on daedalus**:

**Stage 1 (≤30 min) — Liveness positive control**
- Single cell V_G1=0.6, V_G2=0.0, recipe = v5b.
- Variant A (control): current v5b.
- Variant B: enable `use_well_diode=True` with `vnwell_Rs=1e8 Ω` (gemini's path — uses already-wired infrastructure).
- Variant C: kill BSIM impact-ionization (`iii_to_body_factor=0`).
- Variant D: kill BJT (`mbjt=0`).
- **PASS structural confirmed if**: B differs from A by ≥1% RMSE AND (C or D) shifts RMSE by ≥0.5 dec.
- **FAIL (parametric only) if**: A=B bit-exact and C,D both move <0.1 dec → no path is live; deeper structural problem than body diode.

**Stage 2 (≤90 min) — Mini BBO conditional on Stage-1 result**
- If structural confirmed: fix `body_pdiode_Rs` properly, then run 5×5 (Bf, K1_LUT_scale) on 3 representative cells (9 fits × ~10min ≈ 90 min).
- Pre-registered success: any combo <1.5 dec at V_G1=0.6.

If Stage 2 still fails to recover ≤1.5 dec, the structural flaw extends beyond the body diode (likely BJT polarity / iii→body sign / Vb node consumption).

---

## Files
- `research_plan/oracle_queries/O58_structural/prompt.md`
- `research_plan/oracle_queries/O58_structural/openai_response.md`
- `research_plan/oracle_queries/O58_structural/gemini_response.md`
- `research_plan/oracle_queries/O58_structural/grok_response.md`

```


=== FILE: z321_pyport_v5b_progress.json (27462 chars) ===
```json
{
  "script": "z321_pyport_v5b_full_recipe",
  "reason": "after_bf_3000__js_sebas_2.44e4",
  "elapsed_s": 5017.109735965729,
  "ablation_so_far": {
    "bf_50__js_1e-6": {
      "bf": 50,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_1e-4": {
      "bf": 50,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_1e-2": {
      "bf": 50,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_1e0": {
      "bf": 50,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_sebas_2.44e4": {
      "bf": 50,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e-6": {
      "bf": 200,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e-4": {
      "bf": 200,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e-2": {
      "bf": 200,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e0": {
      "bf": 200,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_sebas_2.44e4": {
      "bf": 200,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e-6": {
      "bf": 500,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e-4": {
      "bf": 500,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e-2": {
      "bf": 500,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e0": {
      "bf": 500,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_sebas_2.44e4": {
      "bf": 500,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e-6": {
      "bf": 1000,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e-4": {
      "bf": 1000,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e-2": {
      "bf": 1000,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e0": {
      "bf": 1000,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_sebas_2.44e4": {
      "bf": 1000,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e-6": {
      "bf": 3000,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e-4": {
      "bf": 3000,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e-2": {
      "bf": 3000,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e0": {
      "bf": 3000,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_sebas_2.44e4": {
      "bf": 3000,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    }
  }
}
```


=== FILE: z324_structural_fix_summary.json (6300 chars) ===
```json
{
  "script": "z324_structural_fix",
  "completed": "partial (Stage 1 V1-V3 finished, V4 killed after 16+ min stuck on solver convergence with bjt area=0)",
  "elapsed_s_at_kill": 1432,
  "device": "cuda",

  "code_changes": {
    "D1_emitter_to_Sint": {
      "file": "nsram/nsram/bsim4_port/nsram_cell_2T.py",
      "edits": [
        "lines 513-525: Vbe = Vb - Vsint (was Vb); add D1-fix comment block",
        "lines 537-552: R_Sint balance gets `- Ie_Q1` term (BJT emitter current INTO Sint per SPICE Ie=-(Ic+Ib) convention)",
        "line ~751: local-base inner Newton: Vbe = Vb_local - Vsint",
        "line ~756: finite-diff probe: Vbe = Vb_local + eps - Vsint",
        "line ~772: post-loop recompute: Vbe = Vb_local - Vsint"
      ],
      "loc_changed_approx": 12,
      "verified": "grep shows: Vbe = Vb - Vsint at line 524; -Ie_Q1 in R_Sint at line 552; D1 fix comments at 520,542,751,772"
    },
    "D2_drop_explicit_diodes": {
      "file": "nsram/nsram/bsim4_port/nsram_cell_2T.py",
      "edits": [
        "line 117: use_well_diode default True -> False",
        "body_pdiode_to default was already 'off' (line 162) - no change needed",
        "Rationale: LTSpice has zero explicit diodes; BSIM4 PTM130bulkNSRAM internal Nwell junction is the only path"
      ],
      "loc_changed_approx": 7,
      "verified": "cfg.use_well_diode default = False; cfg.body_pdiode_to default = 'off' (runtime check OK)"
    },
    "D9_Bf_from_card": {
      "file": "nsram/nsram/bsim4_port/bjt.py (no edit needed)",
      "edits": [
        "GummelPoonNPN.from_sebas_card() already returns Bf=10000.0 (line 56)",
        "Dataclass default Bf=10000.0 (line 26)",
        "Card parasiticBJT.txt: bf=10000",
        "z324 honors default — no Bf override in run_branch except passing 10000.0 explicitly"
      ],
      "loc_changed_approx": 0,
      "verified": "runtime check: bjt.Bf = 10000.0"
    }
  },

  "stage1_variants_at_VG1_0_6": {
    "V1_control_D1_D2_D9": {
      "median_log_rmse": 3.248,
      "elapsed_s": 130.7,
      "comment": "Cell-wide V_G1=0.6 median after D1+D2+D9 applied; defaults from class."
    },
    "V2_plus_well_diode": {
      "median_log_rmse": 3.248,
      "elapsed_s": 129.5,
      "comment": "use_well_diode=True, vnwell_Rs=1e8. Identical to V1 (bitwise)."
    },
    "V3_plus_iii_kill": {
      "median_log_rmse": 3.248,
      "elapsed_s": 130.6,
      "comment": "ALPHA0 -> 1e-20 (kill BSIM4 impact ionization). Identical to V1 (bitwise)."
    },
    "V4_plus_bjt_kill": {
      "median_log_rmse": null,
      "status": "killed_stuck",
      "elapsed_s_when_killed": 1432,
      "comment": "BJT area=0. Solver stuck >15 min; killed before completion. Likely degenerate BJT current (zero division-like) caused Newton to fail to converge. V4 inconclusive."
    }
  },

  "stage1_verdict": {
    "v2_vs_v1_shift_pct": 0.0,
    "v2_vs_v1_bitwise_identical": true,
    "v3_vs_v1_shift_dec": 0.0,
    "v4_shift_dec": "inconclusive (killed)",
    "structure_confirmed": false,
    "still_dead": true,
    "diagnosis": "Per gate criteria: V2=V1 bitwise (no well-diode effect) AND V3 shifts <0.1 dec (no impact-ionization effect). The body-voltage path remains structurally inert. D1 rewiring of the BJT emitter to Sint does NOT, by itself, make the body branch live. Adding well-diode on top of D1 makes ZERO difference. Killing impact-ionization makes ZERO difference. This means the current at V_G1=0.6 is essentially 100% BSIM4 channel Ids (M1 in saturation feeding M2 in subthreshold/triode), with Vb and Vsint at the trivial Newton fixed point where all body-side currents cancel near zero. The BJT never fires because Vb never rises (even with E=Sint, both Vb≈0 and Vsint≈0 → Vbe≈0)."
  },

  "stage2_BBO": {
    "status": "SKIPPED",
    "reason": "Stage 1 verdict: STILL DEAD (not STRUCTURE_CONFIRMED). Per task spec, Stage 2 BBO is conditional on Stage 1 liveness confirmation. Running BBO over a dead path would consume budget without producing structurally meaningful results."
  },

  "comparison": {
    "z304_baseline_cellwide_median": 0.99,
    "v5b_cellwide_median": 3.01,
    "z324_V_G1_0_6_median": 3.248,
    "z324_cellwide_estimate": "not_measured (only V_G1=0.6 branch ran)",
    "delta_vs_v5b_at_VG1_0_6": "+0.24 dec WORSE than v5b at V_G1=0.6",
    "delta_vs_z304": "+2.26 dec worse"
  },

  "locked_gate": {
    "criterion": "cell-wide median < 1.5 dec",
    "result": "FAIL (V_G1=0.6 alone is 3.248 dec; cell-wide will be similar or worse)"
  },

  "pitfalls_and_findings": [
    "D1 (emitter -> Sint) by itself does NOT activate the body-voltage path. With E=Sint and Vb≈Vsint at the operating point, Vbe stays near zero and the parasitic BJT never conducts.",
    "D2 (drop explicit well_diode) removed a redundant path but the remaining BSIM4 internal junction is also not contributing meaningfully at V_G1=0.6 (verified: enabling explicit well_diode in V2 gave bitwise-identical result).",
    "D9 (Bf=10000) had no effect because the BJT never fires.",
    "The structural fault is deeper than the audit hypothesized: even with the correct topology, the body node is sitting at a quiet KCL fixed point where ALL secondary paths (well_diode, impact-ionization, BJT) cancel to numerical zero. The current is dominated by BSIM4 channel transport.",
    "V4 (bjt_kill via area=0) caused solver divergence — area=0 introduces a singular Jacobian column in the inner Newton. Should have used `cfg.use_bjt=False` instead.",
    "Implication: the 'dead branch' is NOT the body p-n diode in isolation, NOR the BJT emitter topology, NOR impact-ionization-to-body. The KCL at the body node converges to a region where the model fundamentally cannot produce the snapback-style increase seen in Sebas's measurement at V_G1=0.6. This is consistent with R_deep_B oracle gpt-5's call: 'z304 was a spurious local optimum on wrong physics; v5b is honest but structurally insufficient.'",
    "Next steps recommended: (a) instrument the solver to log Vb, Vsint, and all KCL components at convergence to identify the actual dominant terms; (b) reconsider whether LTSpice .asc truly has no body-bias source (look for an .ic or .nodeset on B); (c) examine whether the TAT path or GIDL are providing any body current at V_G1=0.6, and if not, what is."
  ]
}

```
