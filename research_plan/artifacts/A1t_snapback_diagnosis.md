# A1t — Snapback Diagnosis: why our Id(Vd) is smooth where Sebas's S-curves

**Date:** 2026-05-01 · **Bias of interest:** WORST (VG1=0.6, VG2=0.0)
**State:** z91g v20 median log-RMSE 0.95 dec; arclength reports `n_folds=0`
on every curve while SPICE produces a sharp snapback fold.

---

## 1. Element-by-element comparison: SPICE deck vs `_residuals`

`2tnsram_simple.asc` (data/sebas_2026_04_22) instantiates exactly four
elements and pulls in two `.inc` model files:

| Sebas `.asc` element | Model | Wiring | Match in `_residuals` (`nsram_cell_2T.py:341+`) |
|---|---|---|---|
| `M1 nmos4` (l=Ln, w=Wn) | `.model NMOS` from `PTM130bulkNSRAM.txt` | D=`Din`, G=`G`, S=`Sint`, B=`B` | `_eval_mosfet(model_M1, …)` at L:369 — ✓ correct topology, with body=Vb |
| `M2 nmos4` (l=Ln·10, w=Wn) | same `NMOS` model | D=`Sint`, G=`G2`, S=`0`, B=`B` | `_eval_mosfet(model_M2, …)` at L:372 — ✓ correct |
| `Q1 npn` (area=1u) | `parasiticBJT.txt` (Is=5e-9, Bf=10000, Br=100, Ikr=0.1, Va=100, Ne=1.5, Re=Rc=0.1) | C=`Din`, B=`B`, **E=`0`** | `compute_bjt(bjt, Vbe=Vb, Vbc=Vb-Vd)` at L:386–388 — ✓ emitter=GND, fixed in A.1.i; `area=1e-6` honoured |
| `C1 cap` (CBpar=1f) | trivial | between `B` and `0` | `cfg.Cbody` (transient only). DC: irrelevant ✓ |
| `.inc PTM130bulkNSRAM.txt` | NMOS+PMOS BSIM4 cards | — | NMOS card is what SPICE actually fits. Our z91g uses `M1_130DNWFB.txt`/`M2_130bulkNSRAM.txt` (ALPHA0/BETA0/JSS values are *the same* as PTM130 line 24-25, so card choice is not the gap) |
| `.inc parasiticBJT.txt` | NPN model | — | `bjt.from_sebas_card()` mirrors verbatim ✓ |

**Sebas's deck has NO `vnwell` source, no PMOS instance, no extra
substrate contact, no .nodeset, no .ic.** Every SPICE element has a
PyTorch counterpart and the wiring matches. The 5–11 decade Id gap is
**not** a missing element in the netlist.

Two elements that exist in our `_residuals` but NOT in Sebas's `.asc`:

1. **`vnwell` deep-N-well diode** (cfg.use_well_diode, L:421–438). Sebas
   applies +2 V externally to the package; his `.asc` ignores it. We
   model it as a Shockley diode with series-R between `vnwell` and node
   `B`. Status: **present, but Rs=1e10 Ω makes it impotent** (see §2.3).
2. **gmin shunts** on B and Sint (L:471–477). Numerical only. Match
   ngspice gmin behaviour at gmin=1e-15.

Conclusion of element comparison: **no missing element**. The fold gap
must come from numerical behaviour of one of the present elements.

---

## 2. Per-candidate verdict

### 2.1 PSCBE (BSIM4 §6.4) — second-order substrate-current body coupling
**VERDICT: ALREADY PRESENT, USER'S FRAMING IS WRONG.**

PSCBE in BSIM4 is the *substrate-current body-effect on output
conductance* — it enters `Va_SCBE` as `1/Va_SCBE = (pscbe2/Leff)·exp(-pscbe1·litl/(Vds-Vdseff))`,
boosting `Ids` in the saturation tail.  It is **not** a separate
"second-order body-charging path".  In BSIM4 the entire body-current
path is `Iii` (§6.1).

- `dc.py:783-796` — PSCBE is implemented and active when `pscbe2>0`.
- M1/M2 cards: `pscbe1=5.331e8, pscbe2=1e-5` → both branches live.
- `_model_card_data.py:532-533` defaults match.

PSCBE does **not** create a body-charging mechanism that BSIM4 hides
elsewhere.  Adding a "PSCBE body-effect term" would not be physical.

### 2.2 BSIM4 junction breakdown (§10.1) — Ibd_breakdown
**VERDICT: NOT IMPLEMENTED, BUT IRRELEVANT FOR THIS BUG.**

`compute_body_diodes` (`diode.py:10-13`) is dioMod=1 only; the breakdown
branch is an explicit `TODO(reverse-bv)`. M1/M2 cards set `BVS=10 V`,
`XJBVS=1`. Breakdown current would only kick in at Vbs ≲ −10 V; our
worst Vbs = Vb−Vsint ≈ −0.5 V.  At those biases breakdown contributes
**zero**.  Skip this candidate.

### 2.3 BJT positive-feedback loop topology + loop gain
**VERDICT: TOPOLOGY CORRECT, LOOP GAIN ALWAYS < 1 BECAUSE Iii ≈ 0.**

Topology check (`nsram_cell_2T.py:386–408`):
- `Vbe = Vb` (emitter at GND) ✓
- `Vbc = Vb − Vd` ✓
- `Ic_Q1` is added to `R_Sint` only **indirectly** (via `Id` at the
  drain pin); it does NOT appear in `R_Sint` (correct: Q1 emitter=GND
  so Ic enters drain pin, not Sint). 

Now the loop:
```
Vb ↑  →  Iii_M1(Vds_M1=Vd-Vsint, Idsa) increases (via exp(-β0/diff))
       →  Vsint ↓ slightly via stronger M1 body-effect
       →  M1 channel Vbs_M1 = Vb−Vsint ↑ → Vth0 lowers → Idsa↑ → Iii↑
       →  positive feedback
```

Per A.1.l (`research_plan/artifacts/A1l_vb_trace.md`), at the **WORST**
bias Iii_M1 = 1.6e-27 A even at converged Vb = −0.253 V; **at the
hypothetical Vb=0.7 it is 5.3e-25 A**, while the BJT base draw at
Vb=0.7 is Ib_Q1 = 2.84e-7 A.  The loop *gain* therefore is dwarfed by
the BJT base load by **18 decades**. No fold can form because the only
positive feedback term (Iii) cannot exceed the linear loss (Ib_Q1).

This is a **physics shortage**, not a topology bug.  Topology is fine.

### 2.4 Continuation solver — does forward sweep miss the post-snap branch?
**VERDICT: SOLVER NOT THE BOTTLENECK.**

A.1.p (`A1p_backward_sweep.md`) ran forward (Vd 0.05→2 V, Vb_init=0)
and backward (Vd 2→0.05, Vb_init=0.85) sweeps with `solve_2t_with_homotopy`.
Result: **max Δlog10 Id = 0.0075 dec, no bistability**.  A.1.g
(`A1g_multiroot.md`) tried Vb_init ∈ {0, 0.5, 0.7, 0.9} — every seed
converged to the *same* root.  `n_folds=0` reported by the arclength
solver correctly diagnoses **the residual surface has a single root**.

A minor coding artefact in `arclength.py:387` and `:593`:
```python
Id = comp.get("Id_total", comp.get("Ids_M1", torch.zeros_like(...)))
```
`comp` does not contain `Id_total` (only `Ids_M1`, `Ic_Q1`, …, see
`nsram_cell_2T.py:479-489`).  The path's `path_Id` therefore equals
`Ids_M1` (BJT collector contribution missing).  This is **cosmetic**
because `forward_2t_arclength_grad` re-solves at every Vd target with
`solve_2t_steady_state`, which assembles `Id = Ids_M1 + Ic_Q1 + Igidl_M1 − Ibd_M1`
(L:784-789).  So the per-bias outputs that z91g consumes are correct.
Worth fixing for diagnostics but it does NOT cause the smooth-ramp.

### 2.5 Element list comparison
Already done in §1. No netlist gap.

### 2.6 Initial-guess discontinuity / predictor failure
**VERDICT: NOT THE PROBLEM, AS PROVEN BY 2.4.**

If predictor failures were silently snapping back to the previous
solution we would see `converged=False` somewhere or duplicate Id
values across consecutive Vd points.  z91g v20 reports `conv=30/30` per
curve and the path is monotone smooth — i.e. *the model itself emits a
smooth Id*. Solver is faithful to a one-branch residual surface.

---

## 3. Recommended fix

### Single most-likely missing physics term

The deep-N-well to P-body forward-bias diode is **already coded** in
`nsram_cell_2T.py:421-438` but is **physically inert at the current
default `vnwell_Rs = 1e10 Ω`**.

Numbers (from A.1.n, with Vb=−0.25 V, vnwell=+2 V):
- V_drive = 2.25 V
- I_diode_unlimited ≈ Js·A · exp(2.25 / 1.017·0.0259) = 1e37 A (clamped)
- I_Rs = V_drive / Rs = 2.25 / 1e10 = **2.25e-10 A**
- BJT base draw at Vb=0.7 V (turn-on) = **2.84e-7 A**  →  ratio 1 : 1260

The series-R-limited well current is **3 decades short of the BJT
base load**, so the body cannot be pulled past Vth0 ≈ 0.34 V, the BJT
never fires, and no fold forms.  Lowering `vnwell_Rs` to **1 kΩ to 1 MΩ**
restores 1 mA – 1 µA of well-body charging — enough to clamp Vb in the
0.7–1.0 V "BJT firing" basin.  At that point Ic_Q1 jumps from 5e-17 A
to ~3 mA in a single Vd step, producing the **6-decade jump** and the
**plateau** (kqb high-injection knee at Ikr=0.1 A) that the data shows.

### Concrete code patch sketch

**File:** `nsram/nsram/bsim4_port/nsram_cell_2T.py`
**Lines:** 126 (single config-default change) + 421-438 (no logic change
required; the term is already wired).

```python
# nsram_cell_2T.py:126 — current
vnwell_Rs: float = 1.0e10

# Patch (try the bracket from A.1.n):
vnwell_Rs: float = 1.0e6   # 1 MΩ — gives ~2 µA at V_drive=2 V, ~7× BJT base load
```

Parameter sweep ranges to try in z91 grid:
| Rs | I_well at V_drive=2 V | Effect |
|---|---|---|
| 1 kΩ  | 2 mA | Likely overshoot — clamps Vb at well diode drop (~1.0 V), Ic saturates at Ikr knee |
| 10 kΩ | 200 µA | Strong fold, may over-shoot Id by 1-2 dec |
| **100 kΩ** | **20 µA** | **Likely sweet spot — 70× BJT base, fold forms cleanly** |
| 1 MΩ | 2 µA | Mild fold; matches Sebas if his measured I_DUT ≈ 2 µA at the plateau |
| 10 MΩ | 0.2 µA | Below BJT base — no fold |
| 1e10 Ω | 0.2 nA | Current default — no effect (status quo) |

Also lift the `vnwell_mbjt` couple (L:135) so VG1=0.2 (where mbjt=0.001
zeroes the well too) actually starts firing.  A.1.n flagged this; for
the snapback families (VG1=0.4/0.6) mbjt=1.0 is already correct so no
change needed there.

A1s (solver robustness) reports tolerance issues at Rs=3e9 due to the
arclength corrector's old `1e-9 A` tolerance vs. ~6.7e-10 A well
current; that fix has shipped (now 1e-13). With Rs=1 MΩ the well
current is 2 µA — well above any tolerance — so the previously
unlocked tolerance regime is irrelevant. Coverage should remain 25/25.

### Falsifiable test (one paragraph)

Set `cfg.vnwell_Rs = 1.0e5` (100 kΩ) only — leave every other model
parameter, including ALPHA0/BETA0/Bf/jss, at the current z91g v20
values. Re-run `forward_2t_arclength_grad` at the diagnostic bias
(VG1=0.6, VG2=0.0, Vd∈[0.05, 2.0] in 40 steps) and dump
`{path_Vd, path_Vb, path_Id, n_folds}`.  **Pass criterion:** (a)
`n_folds ≥ 1` (arclength reports a fold), (b) `Vb` jumps from <0.4 V
to >0.7 V across two consecutive Vd points, and (c) `log10(Id)` sweeps
through ≥4 decades within ΔVd ≤ 0.2 V.  **Fail criterion:** monotone
smooth Id with `n_folds=0`.  If it fails despite Rs=100 kΩ, the
hypothesis is wrong and the gap really does require an extra physical
mechanism (e.g., a drain-substrate Zener feeding Vb that BSIM4 omits).
Implementation effort: **single-line config edit + one re-run**, ~5
minutes wall-clock for the single-bias test, ~30 minutes for the full
25-curve z91g re-validation.

---

## Summary

The model has no missing physics terms in the BSIM4 sense. The "missing
piece" is a parameter-value error: the deep-N-well/P-body diode that
the team correctly identified and added in A.1.n is currently inert
because `vnwell_Rs = 1e10 Ω` chokes the well-side current 1000×
smaller than the BJT base draw at Vb=0.7. Lowering `vnwell_Rs` to
~1e5–1e6 Ω restores the body-charging path, lets Vb climb past the
NPN turn-on knee, and the existing Gummel-Poon (Bf=10000, Ikr=0.1)
provides the snapback fold automatically (Ic blows up to ~mA, then
saturates at the Ikr high-injection knee → plateau). Topology, BSIM4
sub-models, BJT, and arclength solver are all internally correct;
no new code paths are required.
