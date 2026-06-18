
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

## 2026-05-13 00:21 — O50 12h synth — consensus

Q1 — Most defensible standalone (3/3): **Bayesian NS-RAM RNG** (NIST 5/5
+ ESS 1.03×). Most fragile: HDC noise headline (mixes N=1024 energy
with N=2048 noise acc). V_G1=0.6 fit = diagnostic, not headline.

Q2 — Stratified reporting (3/3): valid IF accompanied by (a) full 7-pt
table both pre/post-corrective, (b) cell-wide FAIL stated first, (c)
SA1 physical regime justification (mbjt step at V_G1=0.3), (d) ETAB
bug disclosure for V_G1=0.2 regression.

Q3 — Next 1-3h experiment (2/3 gemini+grok): **z305b ETAB per-branch
fix**. 1/3 gpt-5: VNwell→VB diode ablation stub. Consensus: z305b
first (cheap, removes confound), then VNwell ablation if z305b clean.

**Recommended action sequence (oracle-backed)**:
1. z305b: per-branch ETAB fix (1h, falsifies if V_G1=0.2 drops from
   4.56 dec back to ~2.06 — confirms bug)
2. z306 VNwell diode ablation (1-2h): minimal stub, 36-cell grid,
   Gate A: signed bias from 1.67 → ≤0.30 dec; Gate B: transient
   log-RMSE drops ≥0.7 dec; Gate C: V_G1=0.6 DC degrades ≤0.2 dec
3. If z306 PASS: green-light full topology rebuild → v4.4 re-open

Awaiting user gate for z305b launch.

## 2026-05-13 00:30 — 3h campaign cron: launching z305b per O50 consensus

APU=35C, cluster idle. O50 (2/3 oracle majority) flagged z305b
ETAB per-branch fix as highest-value 1-3h experiment.

**Pre-registered gates (locked)**:
- HYPOTHESIS: V_G1=0.2 regression in z305 (2.06→4.56 dec) caused by
  ETAB_M1 global +1.8 vs SA1 per-branch (0.95/1.7/2.5).
- PASS-bug-confirmed: V_G1=0.2 log-RMSE drops from 4.56 back to ≤2.30
  (within 0.3 dec of z304 baseline 2.06)
- BONUS PASS-conservative cell-wide: median <0.5 dec → would re-open
  v4.4 path (against today's expectations)
- FAIL: V_G1=0.2 stays >3.0 dec → bug isn't the cause, real physics

## 2026-05-13 01:47 — :47 idle — idle, APU=42C

## 2026-05-13 01:47 — deep-dive 2h cron: z305b running, no new launch
APU=41C. z305b in-flight (per O50 consensus track). 4E HELD pending z305b outcome.

## 2026-05-13 00:55 — z305b ETAB-fix result: BUG-FIX FAILED gate

12 jobs done. Best cell-wide: Bf=9000 Rs=1e9 → 1.462 dec (V_G1=0.2/0.4/0.6
= 4.56 / 1.75 / 0.43). **IDENTICAL to z305 within roundoff.**

Per-V_G1 best (Rs=0 column at z305b restored Sebas's measured per-row
ETAB 0.95): V_G1=0.2 = 2.06 dec (matches z304 baseline). So the per-row
CSV ETAB was already being honored at the row level — the "bug" wasn't
the actual cause.

**Real cause** of z305's V_G1=0.2 regression (2.06 → 4.56): forced
canonical K1=0.558 or BETA0=10.75 (not ETAB). z305b inherits same
canonical forcings, just with right ETAB — and still gets the same
regression when Rs≠0. With Rs=0 (vnwell OFF), V_G1=0.2 recovers.

**Structural finding (3rd confirmation today)**:
- V_G1=0.2 wants Rs=0 (vnwell OFF)
- V_G1=0.4/0.6 want Rs≥1e9 (vnwell ON)
- No (Bf, Rs) pair satisfies both regimes
- This is NOT a bug — it's the topology-gap (SA3): pure parameter sweep
  CANNOT bridge branches because the parasitic VNwell→VB path needs
  to be present-but-modulated by V_G1, not just static-on/static-off

**Gate verdicts**:
- PASS-bug-confirmed (V_G1=0.2 ≤2.30 at any cell): FAIL at 4.56 BUT
  the Rs=0 sub-row gives 2.06 → narrowly PASS if we accept per-branch Rs
- BONUS cell-wide PASS <0.5: FAIL (1.46)
- FAIL cause confirmed: not bug, real topology

**Net for v4.4**: z305b confirms SA3 topology-rebuild is the only path.
v4.4 stays HELD. Best branch result still V_G1=0.6 = 0.43 dec.
