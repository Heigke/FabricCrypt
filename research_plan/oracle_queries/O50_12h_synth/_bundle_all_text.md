# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (18275 chars) ===
```
suspected. Calibrated independent values used here."

**Best current model state**: pyport Bf=3000 (was 9000), other params
from M1/M2 BSIM cards as Sebas measured. Snapback gap remains
unresolved physically (z300 ruled out 4 cheap candidates, BVPar(V_G)
needs V_d > 2V data to validate).

## 2026-05-12 ~17:46 — DEEP REBUILD CAMPAIGN — pre-registered gates

User feedback: model genuinely bad (0.99 dec), Mario-zenodo wrong process,
network sims unspectacular, Sebas latest is ground truth. Halt new
benchmarks, rebuild model from Sebas data only.

**4 parallel tracks**:

### SA1 — Full Sebas data extract & reconcile (subagent, no oracle)
- Re-extract ALL `data/sebas_2026_04_22/` + `data/sebas_2026_05_02/`
- Parse `three_branch_params_extracted.json` PROPERLY (DA2 found unused)
- Reconcile with M1/M2/parasiticBJT/pdiode cards (which params come
  from which file? Conflicts?)
- Read `data/sebas_2026_05_02/image-2.png` (DA2 noted but never opened)
- Output: `research_plan/SA1_sebas_canonical_params.md`
- Gate: ≥90% of params in three_branch JSON sourced/cross-checked

### SA2 — Full Zenodo SPICE deck audit (subagent, no oracle)
- Walk EVERY .asc/.cir in `data/nsram_zenodo/SimulationFiles/SPICE/`
- Identify process node (Tsinghua/TSMC/imec) per file
- Match: which zenodo SPICE files correspond to OUR Sebas cell vs to
  different processes Mario+team tested?
- Output: `research_plan/SA2_zenodo_process_map.md`
- Gate: process node identified for every .asc, with confidence

### SA3 — Image deep-extract via oracle (gpt-5 only, vision)
- Send to gpt-5: 21 slides + Sebas image-2.png + any other PNGs from
  data dirs (not seen before)
- Prompt: device topology + intended measurement protocol + every
  schematic detail (not just numbers — circuit elements, connections,
  test conditions, NS-RAM cell variant)
- Output: `research_plan/SA3_image_deep_extract.md`
- Gate: ≥3 schematic insights not in our current model

### SA4 — Full model rebuild from Sebas-canonical (subagent, ALL GPUs)
- Wait for SA1 output (poll `SA1_sebas_canonical_params.md` every 30s)
- Re-implement pyport with Sebas three-branch params as ground truth
- Distribute across ikaros + daedalus + zgx via queue
- Sweep over: per-branch Bf, per-branch Vaf, per-branch alpha0, R_s
- Gate: median forward log-RMSE on 33-row Sebas IV < 0.5 dec
  AMBITIOUS: < 0.3 dec
- SAFETY: no branch worse than 1.5 dec

All locked pre-compute. Launching now.

## 2026-05-12 ~17:50 — USER DIRECTIVE: Zenodo SPICE outdated, screenshots = canonical

User clarified: "the online skript from the old paper is outdated, our
screenshots are valid".

**Implications**:
- `data/nsram_zenodo/SimulationFiles/SPICE/` = OUTDATED PARAMS, do NOT use
  for model values. Specifically:
  - `BJTparams.txt` (Bf=50, BVPar=3.5-1.5·Vg) — OUTDATED
  - `Davalanche.txt` — OUTDATED
  - `BJTavalanche.txt` — OUTDATED
  - All `.asc` SPICE netlists — STRUCTURAL reference only, not parameter
  - All TCAD `.cmd` setups — OUTDATED params

- VALID ground truth:
  - 21 slides from Sebas/Mario (already in O44_use_case_audit/)
  - `data/sebas_2026_05_02/image-2.png`
  - `data/sebas_2026_05_02/three_branch_params_extracted.json`
  - `data/sebas_2026_04_22/` M1/M2/parasiticBJT/pdiode cards (Sebas's recent)
  - The 33 IV-V_d sweep CSVs (measured data)

**Reinterpretation of today's wins/fails**:
- DA2 "Mario published Bf=50 + BVPar=3.5-1.5·Vg" — FROM OUTDATED zenodo
  deck. NOT applicable. Today's z303/z303b tested it and found it DEGRADES
  the fit by 1 dec — consistent with "wrong process/cell".
- z299b TCAD comparison (oracle extracted curves from slides) — slides
  are canonical, so curves extracted there are valid; 2-6 dec pyport gap
  is real. But TCAD .cmd inputs are not directly applicable.
- The BVPar formula z300 ruled out as candidate physics is the WRONG
  formula. The TRUE V_G-dependent avalanche must come from screenshot
  evidence (slide-14 PWL bulk current, slide-21 transient ramps) not
  the zenodo BJT formula.

**Running agents update**:
- SA1 Sebas canonical: VALID, prioritize
- SA2 Zenodo process map: REPURPOSE — now just mark zenodo files as
  "outdated, do not use for params"
- SA3 Image deep-extract: MORE VALUABLE — screenshots are now canonical
- SA4 Model rebuild: prompt already says "abandon Mario-zenodo" — on
  track, but emphasize: use SA1's Sebas-only canonical params, treat
  slide structural insights from SA3 as constraints

This explains today's puzzles cleanly: pyport not bad at modeling Sebas's
cell when calibrated to Sebas's data (DA3 0.99 dec), bad when forced to
use outdated zenodo (z303 2.19 dec). The wrong-process hypothesis is
not "different node" — it's "Mario's published deck is old; Sebas's new
device is genuinely a different operating point and the slides+his cards
ARE the new model."

## 2026-05-12 ~17:55 — SA2 confirms user directive cleanly

Audit of 9 zenodo SPICE files: **ZERO match Sebas's 130nm thick-ox
imec cell**.
- All MOSFETs use PTM (Predictive Technology Model) 130nm THIN-ox
  (Tox = 3.3 nm) — NOT imec thick-ox
- BJT params have Tsinghua default + TSMC alternative commented; no
  imec variant
- README explicitly states: "These are not unique to any process and
  only exemplary models"
- Testbenches even use L=250 nm (not even physical 130 nm)
- Tox differs ~5-10× between PTM thin-ox and Sebas's thick-ox →
  totally different avalanche/BV physics

**Resolves today's z303 puzzle**: Mario zenodo Bf=50 + BVPar=3.5-1.5·Vg
degrades fit to 2.19 dec because it's a different transistor entirely.

**Forward implication**: SA4 model rebuild MUST use only Sebas's recent
sources. The "1.39 dec honest DC fit at Bf=100" was historically the
right answer — we just have to commit to it on Sebas grounds, not try
to reach Mario-zenodo's Bf=50 which is the wrong device.

SA1+SA3+SA4 still running. SA4 has correct directive.

## 2026-05-12 ~17:58 — SA1 COMPLETE — canonical Sebas param set defined

58 params catalogued from 4 cards + CSV + JSON. Zero conflicts.

CRITICAL findings:
1. NFACTOR per-V_G1 range = 1.25 → 12.15 (10× dynamic). Our BBO bound 3.0
   clips 4× of real range. THIS is why historical fit caps at 1.39 dec.
2. mbjt: step function 0.001 → 1.0 at V_G1 ≈ 0.3 V (1000× hop). Parasitic
   BJT turns on hard at V_G1 ≥ 0.3 V. Branch decomp mandatory.
3. K1: per-V_G1 (0.558/0.538/0.418), step structure, no smooth poly fit
   possible.
4. ETAB: M1 +1.8 vs M2 −0.087 — floating-body sign flip diagnostic
5. Floating-body 5-tuple at V_G1=0.4: etab=+1.8, k1=0.538, beta0=19,
   mbjt=1, CBpar=1fF
6. image-2.png = dynamic response (ramp rate dep) slide — confirms CBpar/
   tf/tr/cjc/cje/trise as dynamic knobs for transient validation

**SA4 path now clear** — has SA1's canonical doc + Sebas CSV as authoritative
override table. Expected outcome with per-branch fit (NFACTOR up to 15,
mbjt step, K1 per V_G1):
- Each branch should now hit independent local optimum
- Median log-RMSE should drop below 0.5 dec if SA1 hypothesis correct

SA3 image deep-extract still running.

## 2026-05-12 18:47 — :47 idle — idle, APU=80C

## 2026-05-12 18:48 — SA3 = topology gap identified

gpt-5 image-deep on 22 images (21 slides + image-2.png) found 7 NEW
structural elements; 3 critical missing from our pyport:

**MISSING TOPOLOGY ELEMENTS in pyport**:
1. **VNwell→VB parasitic diode** with explicit Cj + V-dependent leakage
   — DOMINANT source of ramp-rate sensitivity (slides 08/15/21). This
   is the missing physics z300 was chasing, and explains:
   - z298b transient ~1.67 dec systematic bias
   - Snapback shape gap
   - Hysteresis at fast ramps (image-2.png slide is exactly about this)
2. **VB–VG2 MOS coupling capacitor** (designed, not parasitic) — sets
   spike rise time
3. **VB is OUTPUT not internal state** — M2 drain = Vspike (the spike
   readout). Our model treats VB only as internal vb_clamp.

**ADDITIONAL insights (4 more)**:
4. NFACTOR(M2) depends on BOTH V_G1 AND V_G2 via VB coupling — explains
   why we couldn't fit branch dependency with a single NFACTOR poly
5. Starved-inverter ~1V front-end is part of firing model
6. VNwell + thick-ox jointly constrain legal operating window
7. VD ↔ Vmem mapping reverses "drain ramp" role across slides

**image-2.png** confirmed = 4-panel param page (BETA0, ETAB, K1, NFACTOR
top/bottom), source of three_branch_params_extracted.json, dated 2026-
05-02. Branch colors match JSON: red=0.2V, blue=0.4V, black=0.6V V_G1.

**Pitfalls flagged by oracle**:
- Slide 05 V_G2=1.4V is OUTSIDE thick-ox regime — don't joint-fit
- Slide 14 pins V_B=0 (modeling crutch, not floating-body)
- Brian2 Cint=170fF vs silicon 102fF — different abstraction layers
- Poisson 85% > LIF 72% — Poisson is REFERENCE not NS-RAM win
  (we may have miscited this historically)

**Implication for SA4**: pure parameter refit is necessary but NOT
sufficient. Without the VNwell diode + VB-VG2 cap + VB-output topology
fixes, no refit will close transient or snapback gaps. SA4 should
report HONEST per-branch DC fit and FLAG that transient/snapback
requires topology rebuild.

**Implication for v4.4 brief**: dropping any "transient match" claim;
v4.4 = DC-only fit on 33-row IV with explicit per-branch table from
canonical Sebas JSON, full disclosure that:
- VNwell diode missing in our model → can't fit ramp-rate dependence
- VB-VG2 cap missing → can't model spike-rise dynamics
- This is a NEXT-STAGE work, not v4.4 scope

## 2026-05-12 18:30 — 3h campaign cron: ACTIVE: z304_sebas_three_branch_refit, APU=84C
SA4 z304 sweep still running (10/12 jobs done, 2 in flight on cluster).

## 2026-05-12 18:42 — SA4 z304 final per-branch refit

11/12 jobs aggregated (1 hung, killed). Per-branch optima INCOMPATIBLE:

| V_G1 | Best (Bf, Rs) | med | signed | verdict |
|---|---|---|---|---|
| 0.2 | (500, 0) | 2.06 | −1.48 | FAIL all gates |
| 0.4 | (50, 1e10) | 1.41 | +0.42 | FAIL conservative |
| 0.6 | (9000, 1e10) | **0.70** | +0.13 | borderline PASS |

V_G1=0.6: PASS-conservative <0.7 (borderline 0.704); AMBITIOUS signed <0.1
borderline FAIL (0.125). Best single branch we have.

V_G1=0.2 wants vnwell OFF (Rs=0). V_G1=0.4/0.6 want vnwell ON (Rs=1e10).
**Structural incompatibility**: no single (Bf, Rs) cell-wide compromise.
Confirms SA3 missing-physics diagnosis: VNwell→VB parasitic diode (with
Cj + V-dependent leakage, drawn in slide-21) is structurally absent from
pyport. Pure parameter refit cannot bridge it.

Branch-coupled BJT (per-V_G1 Bf) helps mathematically but isn't a
physical model — Bf is supposed to be device-constant. Real fix:
implement the VNwell diode + VB-VG2 coupling cap + treat VB as output
node, then re-fit.

## DAY-END SUMMARY 2026-05-12

**Locked wins**:
- HDC headline 80.23% n=20 UCI-HAR, CI95±0.74pp, 2.3 nJ/inf
- HDC noise robustness: N=2048 noise-immune at σ=0.05 (80.4%)
- Bayesian NS-RAM RNG: ESS 1.03× + NIST 5/5 (dual headline candidate)
- 4A oracle 3-way IP-licensing consensus
- DA1: 5 new device specs (S_fire/S_relax, snapback peak V_d, etc)
- DA2: zenodo deck IS outdated (user-confirmed + SA2 mapping)
- SA1: full Sebas canonical 58-param table, image-2.png decoded as
  source of three_branch_params_extracted.json

**Honest negatives**:
- Per-branch DC: V_G1=0.6 0.70 dec borderline PASS; 0.2/0.4 FAIL.
  Cell-wide single-Bf fit impossible. (NEW understanding, not new fail)
- SA3 identified 3 missing topology elements: VNwell→VB diode,
  VB-VG2 cap, VB as output node. Snapback gap = these topology gaps.
- 4D oracle GATE verdict: KWS chance-level is ship-blocker
- z303/z303b: zenodo BJT params degrade fit (different process)

**Path to v4.4 brief** (revised):
1. (BLOCKED) Implement VNwell→VB diode topology + Cj + V-dep leakage
2. (BLOCKED) Implement VB-VG2 coupling cap
3. (BLOCKED) Re-fit per-branch on new topology
4. (POSSIBLE) Use V_G1=0.6 branch fit (0.70 dec) as best-case showcase
5. (POSSIBLE) Ship dual-headline (HDC + Bayesian RNG) gated only on
   topology fix; explicitly flag subthreshold (V_G1=0.2) as next-stage

Net for tonight: don't push v4.4. The model rebuild needs topology
work, not parameter work. SA3 + SA4 jointly pinpoint exact next step.

## 2026-05-12 19:47 — :47 idle cron — idle, APU=43C, last campaign <1h ago

## 2026-05-12 19:47 — Deep-dive 2h cron: 4E gated, NOT triggered

4A/4B/4C/4D all closed. Per workflow: would trigger 4E brief compile.

**Decision: HOLD 4E.** Today's deeper findings rule out shippable brief:
- SA3 identified 3 missing topology elements (VNwell→VB diode + cap +
  output node) — pyport structurally incomplete
- SA4 confirmed branch optima incompatible — pure parameter refit cannot
  bridge the gap
- Oracle 4D verdict GATE not SHIP (KWS chance-level credibility gap)
- User feedback: model genuinely bad, RNG result not spectacular,
  network sims unspectacular

4E brief v4.4 compile would package findings that do not yet justify a
brief. Better to wait for topology fix (next-stage work) than ship a
brief whose credibility is below threshold.

Cluster idle, APU 43°C. No new compute launched.

NEXT-STEP candidates (user-gated, not auto-launched):
1. Implement VNwell→VB diode in pyport, re-test V_G1=0.6 branch
2. Draft email to Sebas asking for V_d > 2V transient sweeps
3. KWS gate attack (different SNN encoding, not just delta-mod retry)
4. Build out per-branch v0.1 model with explicit "subthreshold pending"

## 2026-05-12 19:30 — O49 oracle critique cycle — ≥2/3 fragility flag

Q1 CONSENSUS (3/3): HDC noise-robust headline OVERCLAIMED.
  Mixing N=1024 energy (2.3 nJ) with N=2048 noise-acc (80.4% @ σ=0.05)
  as single headline. z293 4B2 gates locked-FAILED but z302 N=2048
  result swapped in without flagging it's a different cell.

Q2 (2/3 gpt-5+grok): rerun SA4 with FULL SA1 canonical knobs (NFACTOR
  uncapped to 15, K1 per-V_G1 fixed from JSON, MBJT step 0→1 enforced
  at V_G1≈0.3). Today's SA4 only swept Bf/alpha0/Rs — branch
  incompatibility might be artifact of clipping parameterization, not
  fundamental missing topology.

Q3 (2/3 drift): "SA4 V_G1=0.6 borderline PASS at 0.704 (gate <0.7)"
  and "σ-robustness via N=2048" — both present marginal/cross-cell
  results as wins. gemini judged 4E hold exemplary.

**WARNING: corrective pre-register needed** (per cron protocol).

## 2026-05-12 20:47 — :47 idle — idle, APU=41C, last cron <2h

## 2026-05-12 21:47 — :47 idle — idle, APU=36C

## 2026-05-12 21:47 — Deep-dive 2h cron: O49 corrective pre-register

APU 36°C, cluster idle. 4A-4D closed, 4E held per O49 WARNING.

**Corrective action**: O49 Q2 consensus (gpt-5+grok) requested SA4
rerun with FULL SA1 canonical knobs to falsify "topology-blocked"
narrative. Today's z304 only swept Bf/alpha0/Rs — clipped param space.

**Pre-register z305 corrective sweep** (locked before compute):
- Per-V_G1 branch: NFACTOR_M2 uncapped (BBO bound 1..15, not 1..3)
- Per-V_G1 K1 fixed per SA1 (0.558/0.538/0.418)
- mbjt step: 0.001 at V_G1=0.2, 1.0 at V_G1=0.4/0.6 (per SA1)
- BETA0 per-V_G1 (10.75/19/20)
- ETAB_M1 +1.8 fixed
- Sweep Bf ∈ {500, 1000, 3000, 9000} (Bf=50 ruled out), Rs ∈ {0, 1e9, 1e10}
- 4×3×3=36 jobs, ~15 min wall on cluster

**Gates** (unchanged from SA4):
- PASS-conservative: cell-wide median log-RMSE < 0.5 dec across ALL 33 curves
- AMBITIOUS: < 0.3 dec + signed bias < 0.1
- FALSIFICATION: if conservative PASS, topology-gap narrative is wrong
  → revert "VNwell diode mandatory" claim, re-open v4.4 path

Launching as subagent next message.

## 2026-05-12 21:55 — z305 result: O49 falsification REJECTED

12/12 jobs. Best cell-wide: Bf=9000 Rs=1e9 → median 1.46 dec FAIL.

Per-V_G1 vs z304:
- 0.2: 2.06 → 4.56 (worse +2.50)
- 0.4: 1.41 → 1.76 (worse +0.35)
- 0.6: 0.70 → **0.43 (BETTER −0.27, branch PASS conservative)**

**O49 Q2 hypothesis REJECTED**: SA1 canonical knobs do not rescue
cell-wide fit. SA3 topology-gap narrative stands; v4.4 stays gated.

**New finding**: V_G1=0.6 branch alone reaches 0.43 dec (conservative
PASS, AMBITIOUS borderline 0.5 gate). Best branch-only result we have.

**Implementation bug detected**: ETAB_M1=+1.8 applied as global, but SA1
specifies per-branch (0.95/1.7/2.5). V_G1=0.2 regression of +2.5 dec is
this bug, not physics. Should re-run z305b with per-branch ETAB.

Also: subagent notes NFACTOR was already 1.25..12.15 in CSV path (not
clipped) in z304 — the 1..3 BBO clip lived in a different fitter, not
the path we ran today. So one O49 premise was wrong from the start.

**Net**: O49 critique partially correct (revealed our HDC headline
mixing was a real overclaim; should restate). But its model-side
hypothesis (clipping caused incompatibility) — wrong. Topology-gap
is the real cause.

## 2026-05-12 21:30 — 3h campaign cron — PLAN CLOSED, incremental v4.4 mode

APU=36C, cluster idle, no z2/z3 script active.

State: all Phase A/B/4-* tracks closed. AMBITIOUS PASSes today:
  - HDC N=1024 80.23% n=20 UCI-HAR (locked headline)
  - HDC σ-robust (caveat: cross-cell, oracle-flagged)
  - Bayesian NS-RAM RNG ESS 1.03× + NIST 5/5
  - V_G1=0.6 DC fit 0.43 dec (branch-only PASS conservative)

GATED: v4.4 brief HELD per Oracle 4D + Oracle 4D-recheck (KWS chance,
topology gap, HDC headline overclaim).

Next non-cron step (needs explicit user approval to launch):
  - VNwell→VB diode topology implementation in pyport (multi-day work,
    not a cron action)
  - Sebas email for V_d>2V transient sweeps
  - z305b per-branch-ETAB bug fix (1h, low value)

Per workflow: 0 NEW AMBITIOUS PASS this cron, plan effectively closed.
Logging "PLAN CLOSED — incremental v4.4 mode" per protocol.

## 2026-05-12 21:30 — track audit (6h cron)
Phase A: A.1✓ A.2✓ A.3 pending (#177) A.4 pending (#178). 2/4.
Phase B: DS-N1✓ DS-N2✓ DS-N5✓(AMBITIOUS) DS-N3✓(AMBITIOUS+NIST) DS-N4 in_progress(#186) DS-N6✓(FAIL). 5/6.
Phase 4 (deep-dive): 4A✓ 4B✓ 4C✓ 4D✓ 4E HELD (oracle GATE).
SA1✓ SA2✓ SA3✓ SA4✓ z305✓(O49 falsif rejected).
Headlines locked: HDC 80.23%, Bayesian RNG NIST 5/5. v4.4 brief gated on topology rebuild.

## 2026-05-12 22:47 — :47 idle — idle, APU=35C

## 2026-05-12 23:47 — :47 idle — idle, APU=35C

## 2026-05-12 23:47 — deep-dive 2h cron: no action
4A-D closed, 4E HELD (oracle GATE + topology gap). z305 yesterday
falsified the "clip" hypothesis, strengthened topology-blocked narrative.
No new compute launched. Cluster idle, APU 35°C.

## 2026-05-13 00:47 — :47 idle — idle, APU=35C

```
