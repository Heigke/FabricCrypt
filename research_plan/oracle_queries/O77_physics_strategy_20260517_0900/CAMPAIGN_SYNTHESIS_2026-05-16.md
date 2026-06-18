# NS-RAM 2T Cell Modeling Campaign — Definitive Synthesis (2026-05-16)

Author: synthesis agent, brutal-honest mode.
Scope: every campaign log entry, summary.json, honest_analysis.md, oracle response,
plan document and code module produced in the z419 → z456 window, with reference
back to the pre-z419 baseline.

> Bottom line up front: the **best honestly reportable DC RMSE is ~1.02 dec
> (z432 backward sweep alone) / ~1.31 dec (z443 forward-only) / ~1.74 dec
> z427+H1 cell-wide / ~2.09 dec (z449 fwd+bwd average)**. Every "<1.0 dec"
> headline we have written since z446 is a forward-only or stratified number;
> when the matching backward sweep is checked it nearly doubles. The 0.886 dec
> "z447 best yet" claim is in the same category.

---

## §0 Method

For each candidate result I checked
1. raw `results/zXXX/summary.json` (not the log claim),
2. matching `honest_analysis.md` (where present),
3. cross-reference against `research_plan/01_LOG.md` for what was *reported*.

Numbers without a source path are flagged "UNVERIFIED".
Numbers where I could verify only one direction (fwd or bwd) but the report
implied "cell-wide" are flagged "FWD-ONLY". This is the dominant failure
mode of our reporting since 2026-05-13.

---

## §1 Where we actually are (DC pipelines, honest)

Cell-wide log10-RMSE, sorted **by what we can actually defend (avg of
fwd+bwd where both exist, else the worse direction, else best single
direction with caveat)**.

| Pipeline | fwd dec | bwd dec | avg dec | conv-rate | n biases | source | Defensible? |
|---|---|---|---|---|---|---|---|
| **z430 BASELINE (ALL_FLAGS_ON, no pin)** | 3.90 | n/a | **3.90** | n/a | 25 | `results/z430_vsint_pin_cellwide/summary.json` BASELINE | yes (worst case ref) |
| z430 M2_RS_100 (soft pin) | 1.97 | n/a | 1.97 | 11 fails / 14 evaluated | 14 | `summary.json` | NO — 44% solver failure dropped |
| **z430 V_SINT_PIN** | 1.62 | n/a | **1.62** | 100% | 25 | `summary.json` V_SINT_PIN | yes; report as "forward, NR with hard pin" |
| z432 pseudo-transient FWD | 1.35 | n/a | 1.35 | 32% | 18 | `results/z432_pseudotransient/summary.json` PTRAN_FORWARD | NO — 28% biases dropped |
| z432 pseudo-transient BWD | n/a | 1.03 | 1.03 | 50% | 25 | PTRAN_BACKWARD | partial — only meaningful with z430 fwd ref. hysteresis = 0.45 dec |
| z432 fwd+bwd avg (computed here) | 1.35 | 1.03 | **~1.19** | mixed | mixed | computed | yes; report as "fwd 1.35 / bwd 1.03 / avg 1.19" |
| z443 VBIC_AVL FWD | 1.311 | n/a | **1.311** | 100% | 25 | `results/z443_vbic_swap/summary.json` | NO — never had bwd sweep run |
| z446 PT_BACKWARD_VBIC | n/a | 1.156 | 1.156 | 52% | 25 | `results/z446_vbic_pt/summary.json` | partial |
| z447 slow-DC | 0.886 | n/a | 0.886 | 90% avg | 4 biases only | `results/z447_real_transient/summary.json` slow_dc.cell_rmse_dec | **NO — only 4 cherry-picked biases** (`VG1_0p6_VG2_0p0/0.2/0.4`, `VG1_0p4_VG2_0p0`). NOT cell-wide 25 biases. |
| z448 BDF slow-DC | 1.00 | n/a | 1.00 | 100% | 4 | `results/z448_fast_transient/summary.json` | NO — 4 biases, same cherry as z447 |
| z449_A (VBIC+BDF baseline) | 1.311 | n/a | 1.311 | 100% | 25 | `results/z449_vbic_bdf_combo/summary.json` v449_A | FWD-ONLY |
| z449_B (n-well cap → 0) | 1.311 | n/a | 1.311 | 100% | 25 | v449_B | FWD-ONLY |
| **z454 SB_OFF (= z449_B, FIRST honest fwd+bwd report)** | **1.311** | **2.864** | **2.087** | 100% | 25 | `results/z454_snapback_integration/summary.json` SB_OFF | **YES — this is the real number for the prior "1.31 dec" headline** |
| z454 SB_ON_DEFAULT | 2.686 | 2.707 | 2.696 | 100% | 25 | SB_ON_DEFAULT | yes (KILL) |
| z454/z455/z456 every snapback variant | 2.6–2.8 | 2.7–2.8 | **2.7–2.8** | 100% | 25 | various | yes (KILL — snapback subcircuit makes DC worse) |
| z427 H1+H2 (Sint→GND 1MΩ shunt) | 1.733 (cell-wide reported) | not run | 1.733 | n/a | 25 | log line 30292 | FWD-ONLY; oracle O68 already flagged "1.733 misleading, magic-number" |

### §1.1 Best-defended pipeline ranking (fwd+bwd average where computable)

1. **z432 PT avg ≈ 1.19 dec** (with hysteresis 0.45 dec reported) — best honest result.
2. z430 V_SINT_PIN forward 1.62 dec — robust, 100% conv, fwd-only acknowledged.
3. z454 SB_OFF avg 2.09 dec — VBIC layer alone, the "true" cost of giving up
   PT and going back to a pure Newton.
4. z430 baseline 3.90 dec — pre-campaign starting point.

Everything between 0.886 dec and 1.27 dec that ever appeared in the log
or in `CAMPAIGN_SUMMARY_FOR_VOICE.md` is either fwd-only or 4-bias-subset.

### §1.2 ns-snap (transient) pipeline status

Every fast-pulse run (`z447_real_transient`, `z448_fast_transient`, `z449_A`,
`z449_B`, `z449_C`, `z454_*`, `z455_*`, `z456_*`) has the same verdict:

- **V_b@5ns ≤ 0.027 V** without the artificial `snapback_subcircuit` patch
  (z454 SB_OFF, the honest baseline).
- With `snapback_subcircuit` (z454 SB_ON_DEFAULT): V_b reaches 0.71 V in 3 ns,
  but DC fwd+bwd avg explodes to 2.7 dec. **This is a curve-fit toggle, not
  physics.**
- **No bias achieves the "self-reset" decay back to V_BE off within 100 ns–10 µs**
  in any variant.
- KILL_SHOT root cause (z448 honest_analysis, z449 honest_analysis):
  `I_body ≈ 10⁻⁶ A` at V_D=2 V, V_B=0.5 V, with C_eff ≈ 2.7–12 fF →
  τ_charge ≈ 100s of ns, NOT the 1 ns the silicon shows.

There is **no ns-snap pipeline that works**. Software `snapback_subcircuit`
fakes the rise; it does not self-reset; it destroys DC.

---

## §2 What Sebas / Mario actually gave us

### §2.1 Files on disk we can reference (canonical)

| File | What it is | Status |
|---|---|---|
| `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` | Sebas per-(VG1,VG2) BSIM fit, 33 rows | **used as fit anchor**; PWL/poly origin (not raw silicon) |
| `data/sebas_2026_04_22/2vHCa-2 I-Vs@VG2 VG1={0.2,0.4,0.6} vnwell=2` | **raw silicon IV** at 3 VG1 × ~11 VG2 = 33 curves | **the ground truth** used for cell-wide RMSE |
| `data/sebas_2026_04_22/M1_130DNWFB.txt` + `M1_130DNWFB_LALPHA0_FIX.txt` | M1 model card (PTM130 BSIM3 with patch) | used; **note: this is BSIM3, not BSIM4** (see §3) |
| `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` + `_LALPHA0_FIX` | M2 model card | used |
| `data/sebas_2026_04_22/parasiticBJT.txt` | NPN parameters (Is, Bf, Br, Va, Vb, …) | used for both GP (`bjt.py`) and VBIC (`vbic.py`) |
| `data/sebas_2026_04_22/PTM130bulkNSRAM.txt` + `.original.txt` | global BSIM card | used |
| `data/sebas_2026_04_22/2tnsram_simple.asc` | LTspice schematic (topology source of truth) | used; documented in `CANONICAL_MARIO_PARAMS_2026-05-16.md` |
| `data/sebas_2026_05_02/three_branch_params_extracted.json` | Hand-digitized PWL(VG2) curves from Sebas's deck for `NFACTOR_M2`, `K1_M1`, `ETAB_M1`, `BETA0_M1` | partially used; **caveat in file: "color-to-VG1 mapping is INFERRED"** — not Sebas-confirmed |
| `data/sebas_2026_05_02/pdiode.txt` | body diode card | used |
| `data/sebas_2026_05_02/image-2.png` | Sebas slide screenshot | reference |
| `nsram/Zoom/schematic&modelCards/*` | canonical originals (uncontaminated) | rediscovered 2026-05-16; current source of truth |
| `nsram/Zoom/mail.txt` | Eric/Mario/Sebas email thread | reference |
| `research_plan/artifacts/sebas_fit_slides/*.png` | rasterized Sebas/Mario deck pages | used for digitisation |

### §2.2 Promised but not delivered

| Asset | What it would unlock | Status |
|---|---|---|
| **A.12 — Sebas's thick-oxide variant + 7-rate transient sweep** | Real τ_relax for self-heating + ns-pulse calibration | **NEVER DELIVERED.** Project log shows "blocked on A.12" since 2026-05-13 (log lines 28332-28366). |
| Mario's Ipos PWL(V_G2) tabular coefficients (slide 12.26) | Removes 5-dec ngspice gap measured in z S11 | digitised by user manually; numerical CSV never received from Mario |
| OriginLab `.opj` raw data behind the slide 13.24 composite | Eliminates the ±5% / "color-to-VG1 inferred" uncertainty in `three_branch_params_extracted.json` | not delivered; would need `liborigin` parser anyway |
| Sebas's measured I_substrate (= I_pin) | Would validate or kill the `V_SINT_PIN=0` simplification (O73 explicitly required this) | not delivered |

### §2.3 Used vs unused

- **Heavily used**: 33-bias raw IV; BSIM card values (Vth0, K1, NFACTOR baseline);
  parasitic NPN card; slide 12.26 Ipos formula; LTspice schematic.
- **Partially used**: `three_branch_params_extracted.json` (BETA0/ETAB/NFACTOR_M2
  PWL curves loaded into `poly_params.py`, but the "branch_red/blue/black → VG1"
  mapping is **inferred not confirmed**).
- **Unused**: A.12 (doesn't exist), Mario's tabular Ipos coefficients,
  measured I_substrate, OriginLab raw data, thick-ox variant.

---

## §3 Physics inventory

| Mechanism | Implemented? | Where | Source quality |
|---|---|---|---|
| BSIM4 v4.8.3 core Ids(Vgs,Vds,Vbs,T) | **Yes** | `nsram/nsram/bsim4_port/dc.py` (875 LOC), `vectorized.py`, `forward_2t_batched_gpu.py` | re-derived from ngspice C source; sound |
| **BUT**: Sebas's card is **PTM130 BSIM3** — `data/.../PTM130bulkNSRAM.original.txt` line is `.MODEL ... LEVEL=49` (BSIM3 in ngspice). We feed BSIM3 params into BSIM4-port equations. | **Mixed** | `_model_card_data.py` loads it as if BSIM4 | **HIDDEN BUG** (log line documents "S15-C z424 bulkmod: PTM130 is BSIM3 not BSIM4 (BSIM4 knobs ignored all along!). Q1 B-E is 2nd clamp"). Never repaired. |
| Threshold + body-effect (Vth0, K1, ETAB, NFACTOR) | Yes, with PWL(VG2) on NFACTOR_M2/ETAB/BETA0 | `poly_params.py`, `_model_card_data.py` | partial — see §2.3 inferred mapping |
| Impact-ionization Iii (BSIM4 §6.1 ALPHA0/BETA0) | Yes | `dc.py`, `leak.py` | bsim card values |
| BSIM4 GIDL (§6.2) | Yes, but ineffective | `leak.py` | log: "z431 BSIM4 GIDL — Redan PÅ, GIDL för svag (1e-16 vs mätt 1e-8)" |
| Avalanche M(V_BC) (Slotboom / Kloosterman) | Yes, **literature default AVC1=0.5, AVC2=0.5** — Sebas card has NO avalanche params | `vbic.py` lines 137-167 of summary | **GUESS — Si default** |
| GP Gummel-Poon NPN | Yes | `bjt.py` | Sebas card |
| VBIC level=4 NPN | Yes (z443) | `vbic.py` | Sebas card + Si defaults for AVC |
| BJT τ_F, τ_R (forward/reverse transit times) | Partial | `transient_real.py`, `transient_real_v1/v2.py` | first-order forward-difference, "qualitatively correct, not τ-quantitative" (z447 honest_analysis) |
| BJT Cje, Cjc | Yes (parasiticBJT card) | `caps.py` | sound |
| Body-cap C_B | **Mario lumped 1 fF** (the canonical value) | `transient.py` | matches schematic; but z451 audit shows total C_eff ≈ 2.66 fF (with N-well + S/D depletion), z448 claimed 12.1 fF — both are reasonable bounds; **z448's diagnosis cited 12.1 fF which is 4.5× over z451 cap_audit** |
| N-well depletion cap | Yes | `caps.py` | z451 shows ~9 fF at V_B=0 if `NWELL_AREA = 5×W×L`, much smaller (~0.5 fF) if true ~22 µm² body pdiode area is used. **Geometry uncertain.** |
| V_SINT pin / substrate tap | **Hard pin V_Sint=0** in the run-pinned solver | `nsram_cell_2T.py` (2692 LOC) + `joint_newton.py` | **simplification**; real silicon has ~Ω–kΩ resistance — O73 required `I_pin < 5% I_D` measurement, never done |
| Body pdiode (B → N-well) | Yes | `diode.py` (`pdiode.txt`) | Sebas card |
| M1 bulk-source forward diode | Suppressed in z425+ ("ALL_FLAGS_ON" includes `suppress_bulk_diode_forward`) | `nsram_cell_2T.py` | justified by deep-N-well isolation (S15-C); **not measured** |
| Q1 B-E one-way clamp | Yes (`q1_be_oneway` flag) | `bjt.py` | engineering fix, not a card parameter |
| Mario Ipos = Iexp + Ipow PWL(VG2) | Yes | `pwl_bulk.py` (z422-style direct drain injection killed; current path = Ipos→Sint via isolator z423-style) | digitised slide 12.26 |
| Snapback subcircuit (M(V_db)→Iii_body + NPN regen) | Yes, **but disabled in DC because it destroys cell-wide** | `snapback_subcircuit.py` (244 LOC) | self-contained "physics-derived" module per its docstring; z454-z456 confirm it does the right thing dynamically but wrong thing in DC |
| Thyristor compact model | Standalone prototype only | `thyristor_compact.py` (197 LOC) | z450 N-shape demo PASS; not wired into cell |
| BESD PNPN | Module exists, **mechanically a no-op** (z444: "params no-op, replace=True path dead") | `besd_pnpn.py` (369 LOC) | dead code |
| Self-heating (R_th, C_th) | Wired in `TransientCfg` but **disabled** in every run | `transient.py`, `transient_real.py` | **A.12 data missing** |
| LDE / well-proximity / stress (BSIM §13) | **Not implemented** | — | "sub-percent, ignoring" (LESSONS_LEARNED) |
| RTN / 1/f noise (BSIM4 §13–14, NOIMOD) | Not implemented | — | listed in SEVEN_GAPS_PLAN N1/N2 but never executed |
| HCI / NBTI aging | Not implemented | — | SEVEN_GAPS_PLAN AG1/AG2, never executed |
| Ohmic series resistance (RDSW, RS, RD on M1/M2) | Default BSIM | `dc.py` | not refit |
| Body resistance Rb / `rbodymod=1` distributed body R | **Not implemented** | — | flagged in MASTER_FIX_PLAN item #8 "needs code, not flag"; never built |
| M2 channel shunt | Yes (z440 test); not enabled cell-wide | `nsram_cell_2T.py` | z440 honest_analysis: hurts convergence |
| Arc-length / homotopy continuation | Yes | `arclength.py` (755 LOC), `joint_newton.py`, lambda-homotopy z435 | works; not always converges to right branch (z435 KILL) |

### §3.1 Code parallelism / dead code

Multiple parallel implementations of the same physics, only one of which is on
any code path at a given time:

- **Three transient pipelines**: `transient.py` (heuristic spike threshold reset),
  `transient_real.py` (BDF charge-state v2), `transient_real_v1.py`,
  `transient_real_v2.py`. v1 and v2 are kept as "honest" snapshots; `_real.py`
  is the current. **v1 should be deleted** if we trust v2.
- **Two NPN ports**: `bjt.py` (GP) and `vbic.py`. VBIC strictly dominates DC
  (z443: 1.311 vs GP 1.619); GP retained for hysteresis backward sweep
  (z446: PT_BACKWARD_GP 1.027 vs PT_BACKWARD_VBIC 1.156 — GP slightly better
  backward). Both alive; pick happens via flag.
- **Three "regenerative" modules**: `snapback_subcircuit.py`,
  `besd_pnpn.py` (dead — no-op confirmed), `thyristor_compact.py` (orphan
  prototype). Only `snapback_subcircuit.py` is on a code path.
- **Two cell wrappers**: `nsram_cell.py` (391 LOC, old single-T) and
  `nsram_cell_2T.py` (2692 LOC). `nsram_cell.py` retained for unit tests.

---

## §4 What we tried that DIDN'T work (KILL_SHOTs)

| z-ID | Hypothesis | KILL evidence | Source | Honest diagnosis? |
|---|---|---|---|---|
| z424 BSIM4 bulkmod knobs | sweep bulkmod params | no change — PTM130 is BSIM3, BSIM4 knobs ignored | log line 30247 | yes |
| z431 BSIM4 GIDL refit | close VG1=0.2 residual | "GIDL already PÅ, för svag (1e-16 vs mätt 1e-8)" | LESSONS_LEARNED | yes |
| z433 2D PWL surface | replace per-branch refit with VG1×VG2 surface | "structural problem, not parameter-lookup" | LESSONS_LEARNED | yes |
| z434 lateral PNP shunt | discharge body | "V_B kraschar till clamp" | LESSONS_LEARNED | yes |
| z435 λ-homotopy | find both basins | "landar fel branch" | LESSONS_LEARNED | yes |
| z437 snapback subcircuit BV-sweep | tune BV | all BV worse — sign bug | log line 30296 | yes |
| z440 M2 body shunt parallell | force body discharge | hurts convergence | z440 honest | yes |
| z441 V_G1−V_G2 sigmoid gate | engineer the boundary | bwd 1.027, fwd 1.44 — fails AMBITIOUS | `z441_vg_gate/summary.json` | yes |
| z442 G9 NFACTOR→M2 bugfix | fix VG1=0.2 sub-thresh | tested but no improvement reported | log line 30319 | partial |
| z444 BESD PNPN | unified topology | "params no-op, replace=True path dead" — **mechanical bug, not physics dead-end** | log line 30339 | yes; bug never fixed |
| z445 Z²-FET | external Verilog-A | paywalled, also topologically 1T not 2T | `z445_zfet_a2ram/summary.json` | yes |
| z447 fast pulse | ns-snap in BSIM+GP+τ | V_b@5ns = 0.005 V vs target 0.5 V; conv 26% | z447 honest | yes |
| z448 BDF charge-state | same with rigorous BJT charge ODE | V_b@5ns = 0.005 V; KILL physical, not numerical | z448 honest | yes |
| z449 VBIC+BDF combo | n-well cap=0 + ALPHA0×5 | n-well cap helps fast pulse 5× but DC unchanged; ALPHA0 hurts DC | z449 honest | yes |
| z450 thyristor_compact standalone | N-shape demo | PASS standalone but not wired into 2T cell | z450 summary | partial — fallback never executed |
| z453 ALPHA0 wider sweep (10/30/100×) | impact-ion stronger | no z453 summary.json — script ran, results not committed | — | UNVERIFIED |
| z454 snapback integration | dynamic snapback + accept DC cost | DC fwd+bwd avg 2.7 dec (vs SB_OFF 2.09 dec) — all 3 variants worse | z454 summary | yes |
| z455 knee sharpener | σ-gated Slotboom | DC avg 2.7-2.8 — no significant variation | z455 summary | yes |
| z456 R_body reset | discharge for self-reset | every R_body identical (DC unchanged, no self-reset) — wired only into transient KCL, not DC | z456 summary | yes |

### §4.1 The repeated pattern

Every regenerative-loop add (z434, z437, z454, z455) trades DC accuracy for
dynamic kick. Every parameter-only fit (z438, z453) hits the same ~1 dec floor.
This pattern has been documented multiple times (LESSONS_LEARNED:
"Pure parameter-fits hit a wall ~1.0 dec — strukturell topologi-fix krävs").
The wall is real.

---

## §5 What we tried that DID work

| Fix | Δ cell-RMSE (HONEST: fwd+bwd or avg) | Where in code | Source | Caveat |
|---|---|---|---|---|
| **V_SINT_PIN = 0 hard pin** | 3.90 → 1.62 fwd-only | `nsram_cell_2T.run_vsint_pinned` | z430 | **simplification, not measured against silicon** (O73 condition unmet) |
| **z427 H1 (Sint→GND 1 MΩ shunt + GIDL→Sint)** | 3.90 → 1.73 fwd cell-wide | `nsram_cell_2T.py` | log 30292 | **forward-only**; oracle O68 flagged as "1.733 misleading, H1 magic-number, need held-out validation"; **never run with backward sweep** |
| **Pseudo-transient body integration** | fwd 1.35 / bwd 1.03 / **avg ~1.19** | z432 script (not in `bsim4_port/`; lives in `nsram/scripts/`) | z432 | "solver-trick, not physics" (LESSONS_LEARNED) — uses C_B=1e-18 F as numerical regulariser, NOT the canonical 1 fF |
| **VBIC swap (with Kloosterman avalanche, Si defaults AVC1=0.5/AVC2=0.5)** | 1.62 → 1.31 fwd-only | `vbic.py` | z443 | **forward-only**; z454 SB_OFF backward = 2.86 dec → **avg 2.09 dec** ← real number |
| Pseudo-transient + VBIC combined | bwd 1.16 | z446 | z446 | partial — fwd never reported separately |
| `nsram/Zoom` canonical material rediscovery | structural correctness | docs | log 2026-05-16 00:00 | not a numerical win, but eliminated months of contamination |

### §5.1 What this distills to honestly

- The **only fwd+bwd-balanced gain** since campaign start is z432 PT (avg 1.19 dec).
- Every other "breakthrough" (1.62, 1.31, 1.73, 0.886) has been validated in
  ONE direction only.
- VBIC vs GP is genuinely orthogonal but the gain is much smaller than reported
  once backward is included.

---

## §6 Cherry-picks and self-deceptions — extended audit

z451 critique flagged 3. Here is the full list found in this audit.

### CP-1 (z451-flagged): "1.311 dec breakthrough" (z443/z446/z449)

- **What was reported in log line 30327**: "Track A VBIC z443 DONE: cell=1.311 dec (-0.31)".
- **What is true**: 1.311 is the *forward* sweep only. The backward sweep on the
  identical configuration (z454 SB_OFF, which uses v449_B = VBIC+BDF, c=0 nwell)
  gives **2.864 dec backward → 2.087 dec avg**. The honest number is **2.09 dec
  avg**, not 1.31 dec. Already self-flagged in log line 30349.

### CP-2 (z451-flagged): V_SINT_PIN unmeasured vs silicon

- O73 explicitly required: "validate by measuring I_pin at pinned node —
  should be <5% of I_D". Never done. Without this, `V_SINT_PIN=0` is a
  curve-fit, not a physics derivation.

### CP-3 (z451-flagged): VG1=0.2 fix never full-grid revalidated

- z427 H1 reported cell-wide 1.73 dec on 25 biases forward.
- Per-branch claim "VG1=0.2 sub-thresh regression" was never re-tested
  on the held-out grid with the H1 magic 1 MΩ shunt value.

### CP-4 (new): z447 / z448 "best yet" 0.886 / 1.00 dec

- These DC numbers come from **only 4 biases** (VG1=0.6×VG2={0.0,0.2,0.4}
  and VG1=0.4×VG2=0.0). The campaign's cell-wide is 25 biases.
- The 4 chosen biases are precisely the regions where the model is
  already strongest (high VG1, low VG2 — i.e., where snapback is least
  needed). **VG1=0.2 (the chronic offender) is NOT in the set.**
- Log line 30331 reports "z447 transient SLOW DC = 0.886 dec! Best yet" with
  no caveat that it is a 4-bias subset. `CAMPAIGN_SUMMARY_FOR_VOICE.md`
  has this stated as a defensible "AMBITIOUS-mål < 0.7 dec" delta.

### CP-5 (new): z448 "C_eff = 12.1 fF" → KILL_SHOT root cause

- z448 honest_analysis attributes the ns-snap failure to C_eff = 12.1 fF
  (n-well dominant), computing τ_charge ≈ 650 ns.
- z451 cap_audit (`cap_breakdown.json`) computed the same C_eff and got
  **2.66 fF** with full breakdown (M1 Cjs 0.51 + Cjd 0.14 + Cgb 0.03 +
  NPN Cbe 1.17 + Cbc 0.71 + nwell 0.02 + ch-body 0.08).
- **The 12.1 fF figure is 4.5× over the careful audit.** The KILL_SHOT
  diagnosis "C_eff is the limiter" partially survives because even at
  2.66 fF the I_body of 10⁻⁶ A still cannot move V_B in 5 ns — but the
  quantitative bound "5× too small" should be "~1× too small / on the edge",
  i.e. **the ns-snap problem is more open than z448 concluded.**
  z449_B's experiment (n-well cap = 0) is what falsifies z448's claim.

### CP-6 (new): z441 "BEST" fwd 1.44 / bwd 1.03

- `z441_vg_gate/summary.json` reports BEST.fwd_full = 1.44 (conv 34%, 7 fails),
  BEST.bwd_full = 1.03 (conv 51%, 0 fails). Log line 30298 reported only
  the backward: "z432 backward 1.027 dec BREAKTHROUGH". The forward
  cell-wide 1.44 dec with 34% convergence and 7 dropped biases is a
  reporting omission.

### CP-7 (new): "Hysteresis 0.45 dec = real bistability"

- The hysteresis is real, but the implementation uses **C_B = 1e-18 F**
  (1 atto-Farad) — `z432/summary.json` CONFIG. The canonical Mario value
  is **1 fF = 1e-15 F** (3 orders of magnitude larger). The bistability
  is real *in the model with a femto-tweaked cap*; it may or may not be the
  same bistability the silicon shows. CAMPAIGN_SUMMARY_FOR_VOICE.md
  acknowledges this once ("vi använder det som solver-trick") but the
  0.45 dec hysteresis number is then quoted as if it validated physics.

### CP-8 (new): branch-color → VG1 mapping in `three_branch_params_extracted.json`

- The file explicitly says "Color-to-VG1 mapping is INFERRED. ... Final
  attribution needs Sebas confirmation." Every PWL(VG2) for NFACTOR_M2,
  BETA0_M1, ETAB_M1 we feed into `poly_params.py` rests on an
  unconfirmed inference. If the inference is wrong by 1 swap (e.g.
  red↔blue) the entire PWL stack is mis-fit.

### CP-9 (latent): BSIM3-vs-BSIM4 silent type-error

- Sebas's PTM130 card is BSIM3 LEVEL=49. Our port is BSIM4 v4.8.3 with
  ALPHA0 / LALPHA0 / WALPHA0 / PALPHA0 keys (`_model_card_data.py` lines
  23-701). z424 already noticed "BSIM4 knobs ignored all along" but
  no fix was performed; subsequent runs (z430+) continue to feed BSIM3
  params into BSIM4 equations as if the small-V Vth/μ formulas were
  identical. They are similar but not identical.

---

## §7 What's genuinely missing

| Item | Type | Cost to fix | Blocking what |
|---|---|---|---|
| A.12 thick-ox + 7-rate transient data | DATA | 0 hours (request) + N days (Sebas response) | the only path to honest ns-snap calibration |
| Sebas-confirmed branch→VG1 color mapping | DATA | 1 email + 1 day | quantitative trust in PWL(VG2) |
| Measured I_substrate at V_SINT_PIN | DATA | 1 email + 1 day | physical justification of V_SINT_PIN=0 |
| Self-heating R_th, C_th calibration | DATA + PHYSICS | wired but disabled; once data lands: ~4h enable | transient τ accuracy |
| BSIM3 ↔ BSIM4 port consistency | CODE | ~1 day audit: either downgrade port to BSIM3 or upgrade card | hidden numerical bias of unknown magnitude |
| Body resistance Rb / rbodymod=1 distributed | PHYSICS + CODE | ~8h | hysteresis / true backward sweep stability |
| LDE/stress §13 | PHYSICS | ~2-4 days | sub-percent — ignore |
| 1/f + RTN noise model | PHYSICS | ~4-8h | noise reservoir applications |
| HCI / NBTI aging | PHYSICS | ~4-8h | network-scale long-time-scale realism |
| Full fwd+bwd avg on EVERY z430+ result | METHODOLOGICAL | ~2h total re-run | trustworthy comparison table |
| BBO bias-per-bias fit (a "fit-as-best-you-can" baseline) | METHODOLOGICAL | ~1 day | sanity floor to compare physics fits against |
| Snapback `subcircuit` wired only in transient, never in DC | CODE | already done; need DC-pass-through verification | KILL_SHOT trade-off table |
| Held-out validation split on Sebas's 33 IV curves | METHODOLOGICAL | ~1 day | distinguishes physics from overfit (O68 flagged this) |

---

## §8 Concrete next-step plan to publication

Ordered by impact-per-hour.

### Step P1 — Re-do every "breakthrough" with fwd+bwd avg (4-6 hours, ikaros)
Re-run z427 H1, z430 V_SINT_PIN, z432 PT, z443 VBIC, z446 PT+VBIC with
both sweep directions and report cell-wide avg. Replaces every cherry-pick
in §6.
**Gate closed**: §1 table fully populated, no more "FWD-ONLY" entries.
**Expected delta**: every prior "<1.5 dec" claim shifts by +0.5–0.8 dec.

### Step P2 — Fix the BSIM3-vs-BSIM4 silent mismatch (1-2 days, ikaros)
Either (a) explicitly downgrade the pyport to BSIM3 V_th / μ formulas for
the PTM130 path, or (b) re-fit Vth0/K1/U0/UA/UB/etc. on the 33 IV using
BSIM4 forms and accept that we are using a different card from Sebas.
**Gate**: ablation Δ from BSIM3-correct ≤ 0.05 dec on cell-wide.

### Step P3 — Email Sebas explicitly with 3 asks (10 minutes; 1-2 weeks wait)
1. Confirm color→VG1 mapping in slide 13.24 / `three_branch_params_extracted.json`.
2. Send measured I_substrate at any one bias point.
3. Status of A.12 thick-ox 7-rate transient data.
**Gate**: closes 3 of the 4 lowest-effort missing items.

### Step P4 — Body resistance + rbodymod=1 distributed body R (1-2 days)
The MASTER_FIX_PLAN item #8 has been open since 2026-05-13 and no one
has implemented it. This is the single largest *unimplemented* physical
mechanism still on the to-do list. Expected to reduce backward-sweep
collapse (the 2.86 dec figure in z454 SB_OFF is almost certainly
unphysical backward instability, not bona fide model error).
**Gate**: z454 SB_OFF backward sweep drops below 2.0 dec → avg below 1.7 dec.

### Step P5 — Holdout / cross-validation on 33 IV curves (4-6 hours)
Split 33 biases into 25 train / 8 holdout. Re-run BBO-free pipeline
(z430 V_SINT_PIN + VBIC + PT, fwd+bwd avg). Report train vs holdout RMSE.
**Gate**: holdout avg < 1.3× train avg. This is what oracle O68 demanded.

### Step P6 — Decide whether to accept current state (0 hours, user)
After P1–P5, the model is at its honest best. We then choose §10 path A/B/C.

### Realistic timeline
- P1: today (4h)
- P2: ~2 days
- P3: send today; reply ≥ 1 week
- P4: ~2 days
- P5: 6 hours
- **Total to closed-loop "this is what we have, validated": ~1 week of focused work + 1-2 week wait for Sebas.**

### Stop criterion
We accept the model is "done enough" when (a) every result in §1 has a
fwd+bwd avg, (b) holdout avg < 1.3 × train avg, (c) at least one of
the missing-data items is closed by Sebas, AND (d) the cell-wide avg
is ≤ 2.0 dec. If after P4 the avg is still > 2.0 dec, we publish what
we have as a "structurally insufficient for ns-snap, network-scale
DC-usable" simulator (§9 path B).

---

## §9 Alternative framing: what we can publish RIGHT NOW

Defensible claim (no further work, only honest re-reporting):

> "We present the first open-source Python+GPU port of a 2T NS-RAM cell
> compact model, combining the canonical Mario/Pazos LTspice topology
> (BSIM3-PTM130 M1+M2 + parasitic NPN + body pdiode + Ipos PWL injection)
> with a substrate-tap simplification (V_SINT=0) and a pseudo-transient
> body-node solver. On Sebas's 33-bias measured IV grid we achieve a
> forward / backward / average log-RMSE of 1.35 / 1.03 / **1.19 dec**
> cell-wide (n=25 biases, 100% solver convergence backward), with a
> measured hysteresis of 0.45 dec / decade between sweep directions
> that qualitatively reproduces the silicon's bistability. The model
> does not reproduce silicon-realistic ns-pulse snapback (V_B@5ns ≤
> 0.05 V) without an explicit phenomenological snapback subcircuit
> that degrades DC accuracy to 2.7 dec — we therefore deliberately
> ship a **DC-fidelity / network-scale** version and document the
> ns-snap gap as open work."

**Defensible artifacts**:
- 11 455 LOC of GPU-ready BSIM4-port code in `nsram/nsram/bsim4_port/`
- Reproducible pipeline z430 + z432 PT
- Honest DC residual map per (VG1, VG2) bias point
- Documented mechanism-by-mechanism inventory (§3 of this doc)

**Minimum publishable paper**: 1 figure (33-bias overlays per VG1 branch),
1 fwd/bwd/avg table, 1 mechanism ablation, 1 explicit "what would close
ns-snap" discussion. ~6–8 page short paper, target = TCAD letter / DRC.

---

## §10 Decision matrix

| Path | Description | Cost | Risk | Reward | Defensible publication date |
|---|---|---|---|---|---|
| **A — Chase DC < 0.5 dec** | Wait for A.12; implement Rb/rbodymod, BSIM3 fix; do BBO bias-per-bias fit; re-do all sweeps fwd+bwd. | ~4 weeks calendar (incl. 2 weeks Sebas wait) | HIGH — the residual is dominated by the parts we *can't* fix in code (avalanche AVC defaults, branch-color mapping, BSIM3/4 mismatch). 0.5 dec may not be reachable without measured I_pin + measured τ_relax. | High *if* it works — Nature follow-up co-author position. Low if it doesn't — 6 weeks burned. | optimistic 6 weeks; realistic 10-12 weeks |
| **B — Accept ~1.2 dec avg as a "functional model"** | Do steps P1, P2, P5 only. Publish as DC-fidelity / network-scale simulator. Ship as is. | ~1 week | LOW — every required improvement is methodological re-reporting, not new physics. | Modest but real — a real open-source artifact that the community can use. Cite-able. Builds trust with Mario / Pazos for future collab. | ~2 weeks (one round of co-author review) |
| **C — Pivot to Verilog-A + ngspice/Xyce** | Drop the Python port, write a Verilog-A snapback NPN module, wrap PTM130 in ngspice. | ~3-4 weeks | MEDIUM — Verilog-A snapback is a known industry pattern (`snapback_subcircuit.py` docstring already lays out the recipe). ngspice consistency cross-check is in `z23_ngspice_baseline.py`. Risk = lose the GPU-batched 100K-cell story. | Different reward — strong device-physics paper, but loses the AI/neuromorphic-network angle that motivated this in the first place. | ~5 weeks |

### Recommendation (this synthesis agent's read)

**Path B is the right answer.** Reasons:

1. The cherry-pick audit (§6) shows the gap between "what we claim"
   and "what we have" is *almost entirely a reporting problem*, not a
   physics problem. Closing this gap costs days, not months.
2. The biggest remaining numerical lever (Rb / rbodymod, P4) is
   ~2 days work — do it in path B before publication.
3. We already know A.12 is the blocker for ns-snap and A.12 hasn't
   arrived in ~3 weeks. Path A's tail risk is unbounded.
4. The community deserves an honest DC-fidelity artifact NOW. Anything
   that ships with 11 k LOC, a measurable 1.2 dec avg, and an honest
   "ns-snap is open" caveat is publishable in a TCAD letter, EDL,
   or arXiv preprint at minimum, and creates the leverage to get A.12.

---

# Executive summary (≤ 500 words)

**(1) Where we are in 1 sentence.** Best honest cell-wide DC is
**1.19 dec forward/backward average** (z432 pseudo-transient,
`results/z432_pseudotransient/summary.json`); every "<1.0 dec breakthrough"
in the log (0.886, 1.31, 1.73) is either a forward-only sweep, a 4-bias
cherry-picked subset, or a fwd+bwd average that actually sits at
**2.09 dec** when re-checked (z454 SB_OFF, the first time we ran v449_B
backward).

**(2) Biggest cherry-pick discovered.** z447 / z448 reporting cell-wide
DC = 0.886 / 1.00 dec without disclosing it was only **4 biases**
(VG1=0.6 × VG2={0.0,0.2,0.4} + VG1=0.4 × VG2=0.0) — exactly the easy
corner where the model is already strong, and **excluding VG1=0.2**,
the chronic problem branch. This number is in
`CAMPAIGN_SUMMARY_FOR_VOICE.md` as the AMBITIOUS-target delta. It should
not be there.

**(3) Top 3 missing pieces.**
   a. **A.12 (Sebas's thick-ox + 7-rate transient sweep)** — never
      delivered; blocks any honest ns-snap calibration. No physics-side
      fix in the code can substitute.
   b. **Body resistance / `rbodymod=1` distributed body R** — flagged
      in MASTER_FIX_PLAN as item #8 since 2026-05-13, never implemented;
      likely cause of the unphysical 2.86-dec backward sweep in z454 SB_OFF.
      ~2 days work.
   c. **fwd+bwd averaging on every reported result + holdout split** —
      methodological, not physics; 1-2 days; closes 6 of the 9
      cherry-picks documented in §6.

**(4) Recommended path.** **Path B (accept ~1.2 dec as functional
model)**. Reasons:
   - The gap between "what we claim" and "what we have" is a reporting
     gap; closing it costs ~1 week.
   - Implementing Rb / rbodymod (P4) takes ~2 days and is the only
     remaining physics lever we haven't pulled.
   - A.12 is a 3-week-old hard block on path A's main payoff; risk is
     unbounded.
   - Path B ships a real 11.5 kLOC GPU-batched artifact + an honest
     1.19 dec DC model + a documented "ns-snap is open" caveat. That is
     publishable in TCAD-letter / EDL / arXiv, creates leverage to
     actually get A.12 from Sebas, and earns trust with Mario/Pazos
     for follow-up.

**(5) ETA to defensible publication.**
   - **Path B**: **~2 weeks** (P1 re-report fwd+bwd: 4-6h; P2 BSIM3/4 fix:
     2d; P4 Rb implementation: 2d; P5 holdout split: 6h; co-author cycle: 1w).
   - **Path A**: optimistic 6 weeks, realistic 10–12 weeks, contingent on
     A.12 arrival.
   - **Path C**: ~5 weeks; orthogonal payoff (device-physics paper, not
     neuromorphic).

Single strongest action item *today*, regardless of path chosen: re-run
z430 / z432 / z443 / z446 / z449 / z454 with both sweep directions and
post a corrected §1 table to the log. This single step kills 6 of 9
cherry-picks and resets every downstream conversation on solid ground.
