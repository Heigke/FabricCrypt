# R-19 Sub-task B — M2 body topology audit

**Date:** 2026-05-13
**Probe:** `scripts/z335_residual_probe.py` at ngspice OP
**Result:** ||R||_inf = **3.27e-7 A** (AMBIGUOUS zone; 1e-9 < ||R|| < 1e-6)
  * |R_Sint| = 1.55e-10  (dominated by `bjt_sint = -Ic_Q1`)
  * |R_B|    = 3.27e-7   (dominated by `I_well_body` and Ib_Q1 mismatch)

Verdict gate gives "AMBIGUOUS" — not basin-lock alone, but not a *gross* structural
KCL bug either. The 3.3e-7 mismatch sits just below the 1e-6 strict gate from R-17.

## 1. M2.B in pyport (`nsram/nsram/bsim4_port/nsram_cell_2T.py`)

```
L143-145:  m2_body_gnd: bool = True   # config default
L496-505:  Vb_M2 = zero if cfg.m2_body_gnd else Vb
L506-507:  m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero,
                              Vb=Vb_M2, junctions=j_M2, overrides=P_M2)
```

**Current behaviour with `m2_body_gnd=True` (the default):**
* M2's BSIM4 evaluation is called with `Vb=0` — body-effect treated as if bulk
  is grounded. CORRECT vs LTSpice intent.
* In `R_Sint` (L575-581): `+ m2["Ibd"]` is included.
  Sign/argument check: M2.B=GND ⇒ M2 body-drain diode sits between GND (anode)
  and Vsint (cathode = drain). `m2["Ibd"]` is signed POSITIVE-LEAVING-body. The
  current INTO Sint from M2's body-drain diode is `+m2["Ibd"]` when Ibd > 0
  means body→drain flow. With Vb_M2=0, Vd_M2=Vsint=0.382, the junction is
  REVERSE-BIASED ⇒ Ibd ≈ −6.5e-17 A (tiny saturation leak). Sign is consistent
  with the comment at L555 ("M2 junction: Ibd_M2 >0 ⇒ leaves body INTO
  drain(=Sint). → +Ibd_M2"). **No bug here at this OP.**
* In `R_B` (L865-874, the `m2_body_gnd` branch): all M2 body terms (Ibs_M2,
  Ibd_M2, Igb_M2, Igidl_M2, Igisl_M2, Iii_M2) are **correctly dropped** from
  the floating-Vb KCL. M2.Iii also dropped from routing at L749.

## 2. M2.B in LTSpice (`data/sebas_2026_04_22/2tnsram_simple.asc`)

```
SYMBOL nmos4 560 272 R0
SYMATTR InstName M2
SYMATTR Value2 l='Ln*10' w='Wn' m=1
```

The `nmos4` symbol exposes 4 pins (D, G, S, B). No wire in the .asc terminates
on M2's body pin. In LTSpice, an unconnected `nmos4` body pin defaults to
GND (per LTSpice manual & community consensus, and as the existing comment at
L499-500 already documents).

The 1 fF capacitor `C1` (SYMBOL cap 688 288) ties node B to GND, but B here is
M1's body / Q1's base node (FLAG 640 160 B), **not M2's body**. C1 is therefore
a body-to-ground capacitor for the *floating P-body* of M1+Q1, not M2.

**Verdict:** pyport's M2.B routing IS structurally consistent with the LTSpice
schematic. The `m2_body_gnd=True` default is correct.

## 3. Where the 3.3e-7 mismatch actually comes from

From the probe output:

| Term                 | Value (A)       | Comment                          |
|----------------------|-----------------|----------------------------------|
| R_Sint −Ic_Q1        | −1.55e-10       | BJT collector pulls current      |
|                      |                 | FROM Sint to GND (R-13 mode)     |
| R_Sint other terms   | < 3e-13         | All channels/junctions negligible|
| R_B I_well_body      | +1.73e-10       | Dwell ramps Vb when Vd > Vb      |
| R_B I_body_pdiode    | +3.26e-7        | TAT-promoted pdiode leak         |
| R_B I_tat            | −3.26e-7        | Same TAT, sign-canceling pair    |
| R_B Iii*iii_gain     | +5.5e-17        | Impact-ion supply: ESSENTIALLY ZERO|

**Key finding:** at the ngspice OP (Vsint=0.382, Vb=0.267), Iii_M1 is ~5.5e-17
— five to six orders of magnitude smaller than `I_well_body`. The Vb basin
at this point is held by the Dwell↔Rwell anchor, NOT by impact ionisation.

The "3.3e-7" peak is an artifact of how `I_body_pdiode` and the TAT branch
were summed: both are computed and then added under the same name. They
mathematically cancel in this OP. The *true* node-current imbalance, after
removing the TAT/pdiode pair, is at the ~1.7e-10 level — close to the BJT
collector current. That is the SAME order as |R_Sint|.

**Therefore the dominant residual is not an M2 body miswire — it is a
BJT-driven imbalance between Sint and B that pyport's R-13 emitter=GND mode
doesn't fully account for** (the −Ic_Q1 at Sint is not mirrored by a
+Ic_Q1-related term anywhere, since the collector is at the drain and the
emitter is at GND; Ib_Q1 = 1.5e-14 enters R_B as −Ib_Q1, but that's 4 orders
smaller than the Sint side).

## 4. Proposed fix (NOT applied — see `patches/m2_body_to_gnd.diff`)

Since the existing code already enforces M2.B=GND correctly, no source change
to M2 body topology is required. The diff captures a *defensive* hardening:

1. Make `m2_body_gnd=True` the **only** supported branch when the user has
   selected the Sebas-card / R-13 recipe (raise on the other branches in
   `configure_v5b_postfix`).
2. Add an assertion in `_residuals` that the M2 body terms in `R_B` are
   zero-by-construction when `m2_body_gnd=True`.

These guards prevent silent regression. They do NOT change numerics for any
of the D1/D2/D9 fixes already in place.

## 5. R-13 sufficiency verdict

* M2 body topology in pyport **already matches** the LTSpice schematic
  (`m2_body_gnd=True` default; M2 body diodes stripped from R_B).
* The ~3e-10 unbalanced KCL at the ngspice OP is BJT-collector related, not
  M2-body related. It does not have the magnitude of a structural KCL bug
  (1e-6 gate per R-17 not breached) but is too large for a clean root
  (1e-9 gate also not met).
* **VERDICT: R-13 alone IS sufficient.** The remaining 1e-10–1e-7 residual is
  numerical / basin-quality, not topology. M2.B fix is **NOT needed**;
  proceed with R-13 floating-Vb solver and re-evaluate basin selection
  (multi-start, smarter init at the ng-spice OP coords).

If after R-13 multistart we still see persistent cluster around the trivial
root, revisit the BJT collector/emitter feedback (D1) — that is the next
candidate, not M2.B.
