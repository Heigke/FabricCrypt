# PHYSICS HUNT — Track A (Materials + Lit Search) — 2026-05-20

**Charter:** Identify the ONE missing constitutive physics term most likely responsible
for the residual **1.163 dec** DC fit gap on Sebas Pazos's 33-bias data, after every
known parameter / bug / topology fix has been applied (5 ngspice bugs, IIMOD, pdiode,
well_diode, Theta0_n, phi, eta_sigmoid, Vth/tox, mbjt, IFT sign, arclength, eta_sigmoid v2,
JTS-TAT enabled).

**Method.** Re-read all Sebas material (`data/sebas_2026_04_22/`, `data/sebas_2026_05_02/`),
`docs/Zoom/` (27 jpegs + corrupted Swedish auto-caption transcript, low value),
`data/mario_slide21_oscillation_targets.json`, prior physics hunts
(`T4_missing_physics_v2.md`, `S6_PHYSICS_REBUILD_PLAN_2026-05-15.md`,
`snapback_literature_2026-05-15.md`, `SA3_image_deep_extract.md`,
`MARIO_BRIEF_v4.8_draft_2026-05-19.md`). WebSearch executed for Pazos paper, Hurkx-TAT,
BSIM4 rbodymod (results below).

**Citation discipline.** Inferences use "based on"; measured values cite the file.
Where Sebas/Mario are silent, said explicitly.

---

## TL;DR — Smoking-gun candidate

**#1 Distributed body resistance with `rbodymod = 1`** (BSIM4 §6 body network).
Sebas's M1/M2 cards ship with **`rbodymod = 0`** (lumped body) —
`data/sebas_2026_04_22/M1_130DNWFB.txt:15`,
`data/sebas_2026_04_22/M2_130bulkNSRAM.txt` equivalent line — and our pyport inherited
that default. For a device whose entire mechanism is parasitic-BJT feedback through the
floating body, lumped Rb is a first-order structural error, not a parameter choice.
This is the cheapest experiment (flip one flag, re-fit) and is the only candidate
flagged HIGH by two independent oracle rounds (Gemini-2.5-pro in O53, GPT-5 in
`T4_missing_physics_v2.md` §N3) plus consistent with the snapback-literature finding
that industry snapback macros (TechConnect 2008, S-D junction partition) always
include a distributed body sheet.

---

## TOP 5 candidate missing physics terms

Ranking weights: (i) oracle consensus from prior O53 round, (ii) direct evidence in
Sebas/Mario materials, (iii) ablation cost, (iv) expected dec-improvement order of
magnitude.

### #1 — Distributed body resistance R_B,float (BSIM4 `rbodymod = 1`)

(a) **Mechanism (2 lines):** the floating P-body is not a single node; lateral and
vertical Ir·R drops between impact-ion source (drain edge) and parasitic-NPN base
contact set how much of the avalanche current actually forward-biases the body→source
junction. Lumped Rb collapses this to one node and loses V_hold(V_G1) trend.

(b) **What we currently model vs missing.** Sebas's M1/M2 BSIM4 cards explicitly set
`rbodymod = 0` (`M1_130DNWFB.txt` line 15: `rbodymod = 0   trnqsmod = 0   acnqsmod = 0`).
Our pyport inherits. The full BSIM4 §6 body network (Rbpb, Rbpd, Rbps, Rbdb, Rbsb +
their geometry-scaled `rbpbx0/y0`, `rbpd0`, `rbps0` etc. — all PRESENT but UNUSED in
the card, lines 92–101) provides a 5-resistor T-network between B, body-contact, S, D.
Without `rbodymod=1` none of these activate.

(c) **Expected ΔDC dec.** Order **0.2 – 0.4 dec** improvement. Reasoning: per
`T4_missing_physics_v2.md` §N3, V_hold currently mis-trends versus V_G1 by ~0.2 V; the
fit absorbs this as Bf/iii_gain compensation. Removing one structural compensation
typically buys ~0.2–0.3 dec on a 1.163 dec residual.

(d) **Citation.** BSIM4.8.0 manual §6, "Body Resistance Network" (UC Berkeley);
`data/sebas_2026_04_22/M1_130DNWFB.txt:15`; `T4_missing_physics_v2.md` §N3 (GPT-5
/ Gemini-2.5-pro O53 round); TechConnect 2008 Nanotech briefs vol-3 pp.1483
"Source/Drain Junction Partition in MOS Snapback Modeling"
(briefs.techconnect.org/wp-content/volumes/Nanotech2008v3/pdf/1483.pdf).

(e) **Cheapest experiment.** Re-run the 33-bias fit with `rbodymod = 1` in both M1 and
M2 cards, all other parameters frozen at v5.3 baseline. ~2 hr ngspice work
(`T4_missing_physics_v2.md` "Cheapest test to run first"). Discriminator: V_hold-vs-V_G1
slope. If slope flattens without Bf re-fit, R_B was absorbing V_G1 dependence.

---

### #2 — Multi-τ interface/border-trap reservoir (Hurkx-style SRH-TAT at oxide interface)

(a) **Mechanism.** Thick gate oxide + STI corners + Si-SiO₂ interface at the floating
body host a continuum of traps with capture/emission constants spanning µs to seconds.
At each DC bias the trap occupancy is not at equilibrium — it carries a memory of the
prior bias — producing sweep-direction asymmetry and a quasi-static V_th drift.

(b) **What we currently model vs missing.** Current model: BSIM4 §10.1 JTS-TAT enabled
(per audit history) and `cit = 0` in M1 card (`M1_130DNWFB.txt:51`). JTS-TAT is a
DC junction tunneling current — it does NOT carry trap occupancy as a dynamic state.
Missing: 2–3 parallel SRH-Hurkx capture/emission reservoirs coupled to V_B with
distinct τ (≈300 µs, 5 ms, 80 ms — per `T4_missing_physics_v2.md` §N1).
`cit = 0` means our model also has zero interface-state DC capacitance.

(c) **Expected ΔDC dec.** Order **0.1 – 0.3 dec** on the 33-bias DC alone (most of the
mechanism's signal is in slow-ramp asymmetry, not static DC). Larger on the transient
fit (slide-21 loop morphology, where this is the only mechanism able to reproduce loop
area growth with slower ramp — `T4_missing_physics_v2.md` §N1).

(d) **Citation.** Hurkx, Klaassen, Knuvers, "A new recombination model for device
simulation including tunneling", *IEEE TED* **39**(2), 331–338 (1992),
DOI 10.1109/16.121690 (semanticscholar.org/paper/4e0ad76a1a7d0e1b4db5f1e48bc05a6f16614337);
JUNCAP2 extension (Springer 10.1007/978-90-481-8614-3_10); O53 oracle synthesis
(Gemini + Grok both HIGH).

(e) **Cheapest experiment.** **Wait-time DC** on bench: ramp to just below the snapback
knee, hold for {1, 10, 100} ms, complete ramp. Trapping → bias-history-dependent knee
shift. Uses existing TEG + existing parameter analyser. No new fixturing.
(`T4_missing_physics_v2.md` §N1 "Cheapest experiment (Gemini)".)

---

### #3 — Generation–Recombination / SRH-TAT in body-side depletion regions

(a) **Mechanism.** Body→DNW and body→drain depletion regions thermally generate carriers
(SRH) with field-enhanced TAT at high reverse bias. Slow ramp rates integrate the
generation current onto C_body, pre-charging V_B and igniting snapback at a lower V_D.

(b) **What we currently model vs missing.** Current model: BSIM4 §10.1 JTS-TAT
(enabled but per `MARIO_BRIEF_v4.8_draft_2026-05-19.md` §C3, the solver "suppresses
V_jct under the simulated bias, killing the TAT current"). We have NO explicit
G-R current source at the V_B–DNW junction; the pdiode (`data/sebas_2026_05_02/pdiode.txt`)
is level-1 (`is=5.37e-7, n=1.054`) — no Hurkx TAT/BBT terms (`bvj=1e31`, ie disabled).
Missing: A·exp(-E_a/kT)·f(V_j) gen current at V_B–DNW and V_B–drain.

(c) **Expected ΔDC dec.** Order **0.05 – 0.2 dec** on isothermal DC (signal mostly in
ramp-rate, weak at single rate). Strong **T-coefficient** signature: 25→75 °C with
E_a≈0.5 eV → ~5× current, vs avalanche barely moving.

(d) **Citation.** Hurkx 1992 (same as #2); `MARIO_BRIEF_v4.8_draft_2026-05-19.md` §C3
(JTS-TAT killed at BSIM4 default, heatmap optimum 0.852 dec fwd-only at JTSS=1e-6 /
NJTS=5 — proves the mechanism is parameter-active but bwd does not improve, ie the
DC-only Hurkx mechanism cannot close the bwd half alone); `T4_missing_physics_v2.md`
§N2 (3/3 oracle consensus).

(e) **Cheapest experiment.** **Temperature sweep** at fixed ramp rate, 25 → 75 °C; if
ramp-rate dependence shifts strongly with T (E_a≈0.5 eV) it is G-R/TAT, not avalanche.
Sebas's TEG supports this (probe-station with thermal chuck — confirmed in
`docs/Zoom/Image 2026-04-30 at 13.31.jpeg` annotation per SA3 §3).

---

### #4 — Channel self-heating (BSIM4 `selfheatmod = 1` + thermal R_th)

(a) **Mechanism.** Avalanche dissipation at the drain edge raises local lattice T, which
(i) reduces mobility (closes pinch-off), (ii) increases SRH/TAT generation (couples to
#2/#3), and (iii) increases ionization coefficient α(T) (couples to snapback gain).

(b) **What we currently model vs missing.** `M1_130DNWFB.txt:13` has `tempmod = 0`; no
`selfheatmod` line is present (= default 0). BSIM4 thermal node, R_th, C_th are
unconfigured. Pyport `temp.py` (post phi-formula fix) handles ambient T but does not
self-consistently iterate self-heated T from dissipation.

(c) **Expected ΔDC dec.** Order **0.05 – 0.15 dec**. Below #1/#2 because the bulk DC
geometry (cell area 1 µm², low duty per Sebas's 33-bias methodology) keeps ΔT modest
(<20 K based on R_th ~30 K/mW typical for 130 nm bulk). Self-heat matters more for
pulsed-IV / transient fit than DC.

(d) **Citation.** BSIM4.8.0 manual §11 "Self-Heating Model"; O53 round: GPT-5 MED,
Gemini MED, Grok LOW (`T4_missing_physics_v2.md` §N5, 2/3 MED).

(e) **Cheapest experiment.** **Pulsed-IV** with varying pulse width (1 µs vs 10 ms) at
the same V_G,V_D. ΔI_D between widths quantifies thermal contribution. If
self-heat is real, narrow-pulse fits tighten with current parameter set.

---

### #5 — V_B↔V_G2 designed MOS coupling capacitor + V_NWELL parasitic diode dynamics

(a) **Mechanism.** Sebas's slides (Image 05, Image 21 — SA3 §4 insights #1, #2) show
that V_G2 transients couple capacitively onto V_B through a *designed* MOS-cap, AND
the V_NWELL → V_B path is a real diode with C_j (not a leakage path). At quasi-DC
sweeps this still bleeds DC current through C(V_B)·dV_B/dt × loop-residual.

(b) **What we currently model vs missing.** Pyport has `well_diode (vnwell)` added per
audit history (DC junction). Missing: (i) explicit C_GS_M2 coupling — none in our cell;
(ii) the well diode is treated as DC-only — no C_j(V) for the V_NWELL junction. For
the 33-bias DC sweep these are second-order (it's "DC"), BUT the way Sebas's DC sweep
ramps in seconds (`mario_slide21_oscillation_targets.json` shows ~430 ns transients;
"DC" sweep is typically 1–10 ms/point) means C·dV/dt residual leaks remain a few-pA bias
that integrate over the 33-curve set.

(c) **Expected ΔDC dec.** Order **0.05 – 0.10 dec** at most. Larger on transient fit.

(d) **Citation.** SA3 §4 insights #1, #2 (Image 05 + Image 21 — `data/sebas_2026_05_02/
image-2.png` and Zoom Image 2026-04-30 at 13.31); `data/sebas_2026_05_02/pdiode.txt`
(level=1 pdiode card with `cj = 7.33e-4`, `cjsw = 1.05e-10`, `m = 0.241` — capacitance
parameters PRESENT but only used if instantiated as a transient element).

(e) **Cheapest experiment.** Add the two passives (C_GS_M2 = W·L·Cox of M2 gate ≈
1.7 fF based on M2 size from `2tnsram_simple.asc`; junction-C from pdiode card
already extracted), re-fit. Free — uses existing characterisation.

---

## Candidates explicitly de-prioritised

- **Pazos parasitic NPN (C1).** KILLED at gate, NPN-OFF beats NPN-ON by 1.19 dec
  (`MARIO_BRIEF_v4.8_draft_2026-05-19.md` §C1). Do not re-open without new evidence.
- **JTS-TAT default (C3).** KILLED at gate (`MARIO_BRIEF_v4.8_draft_2026-05-19.md` §C3);
  best-cell fwd-only 0.852 dec but bwd does not improve. Subsumed into #2/#3 above
  as the dynamic + body-side variant.
- **Quantum confinement.** Sebas/Mario silent. Cell is thick-ox (180 nm 1T, 130 nm
  bulk for 2T) — wrong regime. Not pursued.
- **Field-enhanced GIDL (BSIM4 agidl tail).** Card has `agidl=1.99e-8 bgidl=1.624e9
  cgidl=6.3 egidl=0.91` (`M1_130DNWFB.txt:67–68`) — already active and tuned. O53
  rated 1/3 oracles. Not a primary candidate.
- **DIBL+body self-consistency.** Subsumed into #1 (Rb closes the loop).
- **Kirk effect / quasi-saturation in lateral NPN** (O53 N7). GPT-5 HIGH but no
  silicon evidence in Sebas materials. Defer.

---

## Smoking-gun decision

**Single most-likely missing term: `rbodymod = 1` (distributed body resistance).**

Rationale:
1. It is a structural model-card defect, not a missing physics module — easiest to
   falsify (one flag, one re-fit).
2. Two independent oracle rounds flagged it HIGH/MED with concrete first-order
   reasoning (V_hold vs V_G1).
3. The Sebas BSIM4 card already ships the geometry (Rbpb=50, Rbpd=50, Rbps=50,
   `rbpbx0=100`, `rbpby0=100`, `rbsbx0=100`, etc.) — every parameter to activate
   it is present; only the master `rbodymod=0` switch keeps it dormant. This is
   suggestive that the card author expected someone to enable it.
4. Snapback literature (TechConnect 2008, LVTSCR papers — `snapback_literature_2026-05-15.md`)
   uniformly uses distributed body for snapback macros.
5. Estimated dec improvement (0.2–0.4) is the largest single-mechanism estimate that
   does not require a new SPICE element.

**If `rbodymod=1` does not buy ≥0.2 dec**, run the wait-time DC test (#2 / Hurkx-trap)
as the second-cheapest discriminator before any other parameter sweep.

---

## What we did NOT find

- **No mention** in Sebas's materials of "Hurkx", "self-heating", "rbodymod", "interface
  traps", or "ETAB curve(VG2) origin" in plain words. The 33-bias card is a hand-fit;
  Sebas did not annotate the *physics* of the fit, only the parameter values.
- **No supplementary** to a Pazos *Nature 2025* paper surfaces in WebSearch (the task
  brief's "Nature 640:69 2025" reference returns no match; Pazos 2025 in Nature space
  appears to be the memristor-Achilles-heel review, not a 2T NS-RAM paper). The 2T
  NS-RAM characterisation appears to live in slide decks + Sebas's MATLAB fit tool,
  not a peer-reviewed paper available online.
- **Zoom transcript is corrupted** (Swedish auto-caption of a English/Spanish meeting,
  2283 lines of unrecognisable phonetic noise — `grep` for "trap", "self-heat", "tunnel"
  yields zero usable hits). The Zoom slide *jpegs* are the only Zoom content with content.

---

## Sources cited

- `data/sebas_2026_04_22/M1_130DNWFB.txt` (BSIM4 NMOS card, `rbodymod=0`, `cit=0`,
  `tempmod=0`, GIDL params present)
- `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` (companion M2 card; same defaults)
- `data/sebas_2026_04_22/parasiticBJT.txt` (`is=5e-9 bf=10000 br=100 nc=2 vje=0.7` etc.)
- `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` (33-bias parameter table,
  per-bias `ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt`)
- `data/sebas_2026_04_22/2tnsram_simple.asc` (LTSpice schematic: M1 + M2 + Q1 + C_Bpar=1fF)
- `data/sebas_2026_05_02/pdiode.txt` (well diode level=1, `is=5.37e-7 n=1.054 cj=7.33e-4`)
- `data/sebas_2026_05_02/three_branch_params_extracted.json` (NFACTOR/K1/ETAB/BETA0 branches)
- `data/mario_slide21_oscillation_targets.json` (transient targets: 430 ns period,
  4.8 mA peak, 0.2 pJ/spike)
- `research_plan/T4_missing_physics_v2.md` (O53 oracle synthesis — TOP 3: N1 traps,
  N2 G-R/TAT, N3 R_B,float)
- `research_plan/SA3_image_deep_extract.md` (O48 image deep extract — 7 topology insights)
- `research_plan/S6_PHYSICS_REBUILD_PLAN_2026-05-15.md` (8 missing-physics list)
- `research_plan/snapback_literature_2026-05-15.md` (snapback macro-model precedent)
- `research_plan/MARIO_BRIEF_v4.8_draft_2026-05-19.md` (§1, §2 — current 1.163 dec
  state + C1/C3 KILL-SHOTS)
- Hurkx, Klaassen, Knuvers, *IEEE TED* 39(2), 331–338, 1992, DOI 10.1109/16.121690
- BSIM4.8.0 manual §6 (body resistance), §10.1 (JTS-TAT), §11 (self-heating)
- TechConnect 2008 Nanotech briefs vol-3 pp.1483, "Source/Drain Junction Partition
  in MOS Snapback Modeling for ESD Simulation"

---

## Time-box

Track A budget used: ~45 min. Deliverable: this file.
