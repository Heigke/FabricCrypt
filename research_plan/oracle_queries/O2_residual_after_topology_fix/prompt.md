# Oracle query O2 — Why does our 2T NS-RAM forward sim under-predict measured Id by 2-5 decades, after the emitter=GND topology fix?

**Recommended model:** GPT-5 Thinking, Gemini 2.5 Pro, or o3 with extended thinking.
This is a physics + circuit problem, not a code-style question. Fast tier
models (gpt-5-nano, gemini-2.5-flash) will not have enough device-physics
depth.

**You will receive:** this prompt + zipped code + Sebas's two foundry
model cards + his per-bias parameter CSV + the failing plot + the LTSpice
schematic + the parasitic NPN model card + Sebas's email history (Apr-17
onwards) so you have the full context.

**What we want:** a ranked diagnosis with concrete tests / code patches
we can run tonight on Ikaros (we have a venv ready). Not philosophical
musings.

---

## State, in a single sentence

A differentiable PyTorch port of BSIM4 v4.8.3 + Gummel-Poon NPN, simulating
Sebastian Pazos's 2T NS-RAM cell, agrees with measurement on shape but is
~2–5 decades low on |Id| in the regime where the parasitic NPN dominates
the firing current — and we want to close that gap.

## What's already been ruled out (don't re-suggest these)

We've run a long chain of diagnostics. The following are eliminated as
the cause of the residual:

1. **NFACTOR override path** — A.1.a confirmed `sd.scaled["nfactor"]=X` reaches the Vgsteff sub-threshold formula correctly; n=12 vs n=1.58 changes Id by 2.7 decades as expected.

2. **mbjt drop** — A.1.b found Sebas's `mbjt` column in his per-bias CSV (1000× scaler at VG1=0.2, 1× at VG1=0.4/0.6) was being silently dropped. Fixed: `bjt.area = csv.area * csv.mbjt` in our `make_bjt`. Verified.

3. **Body-source diode** — A.1.c confirmed `compute_body_diodes` correctly evaluates Ibs/Ibd; not the issue.

4. **Iii formula form** — A.1.d confirmed our formula matches BSIM4 v4.8.3 §6.1: `Iii = (alpha0 + alpha1·Leff)/Leff · (Vds − Vdseff) · exp(−beta0/(Vds − Vdseff)) · Idsa`. With Sebas's BETA0=20 and (Vds − Vdseff) ≈ 0.27 V on M2 at the diagnostic bias, the exponential factor is exp(−74) ≈ 1e-32, killing Iii_M2. **This is what BSIM4 §6.1 actually predicts.**

5. **GIDL=0 at this bias** — A.1.e confirmed: drain-side band-bending gate (Vd-Vg-egidl > 0) is closed for all four edges (M2/M1 × GIDL/GISL). Honest physics, not a bug.

6. **Multi-root convergence failure** — A.1.g: forced Vb_init = 0.0, 0.5, 0.7, 0.9 V; arclength continuation; all converge to the same Vb ≈ 0.34 V root in 3 Newton iters. n_folds=0 from arclength — system has only one root.

7. **IIMOD branch mismatch** — A.1.h: Sebas's two cards declare `version=4.5`, predates BSIM4's IIMOD parameter (v4.7+). Both his and our default IIMOD=0. Same formula.

8. **Parser dropping `.param` continuation lines** — earlier finding: M2 card's `+` continuation lines on `.param` were silently dropped, leaving vth0/vsat at BSIM4 defaults. Post-load patch applied with the right values; verified `model.get('vth0') = 0.54153`.

## The fix that did move the needle (4 hours ago)

A.1.i decoded Sebas's LTSpice schematic (`2tnsram_simple.asc`, attached).
The schematic's parasitic NPN is wired with **emitter=GND, not emitter=Sint**. Our
wrapper had `Vbe = Vb − Vsint`. Sebas's schematic implies `Vbe = Vb − 0
= Vb`. Single line change in `nsram_cell_2T.py:_residuals`:

```python
# BEFORE (wrong)
Vbe = Vb - Vsint
Vbc = Vb - Vd

# AFTER (matches schematic)
Vbe = Vb              # emitter is GND
Vbc = Vb - Vd
```

This dropped median log-RMSE from 4.23 → 1.63 (Newton+homotopy) or 2.03 (arclength, more honest). p90 from 5.83 → 3.89 / 5.03.

## Where we are now

**Median log-RMSE = 2.03 decades, p90 = 5.03.** See `fit_vs_meas.png`.

Per-curve pattern:

| VG1 | VG2 | log-RMSE | conv |
|----:|----:|----:|----:|
| 0.20 | -0.20 | 1.48 | 30/30 |
| 0.20 | +0.10 | 0.43 | 29/30 |
| 0.40 | 0.00  | **3.92** | 28/30 |
| 0.40 | +0.30 | 0.85 | 26/30 |
| 0.60 | 0.00  | **5.73** | 30/30 |
| 0.60 | +0.50 | 1.15 | 30/30 |

**Pattern:** error grows toward low-VG2 / high-VG1, where the parasitic
NPN should fire hard. Our prediction shape is now correct (snapback knee
visible) but magnitude is short.

**At the worst bias (VG1=0.6, VG2=0.0, Vd=1.5):**

- Measured Id ≈ 2 × 10⁻⁵ A
- Predicted Id ≈ 3 × 10⁻¹¹ A (~6 decades short, despite NPN being wired correctly)
- Converged Vsint ≈ 0.31 V, Vb ≈ 0.34 V (both come from `solve_2t_with_homotopy`)
- Vbe = Vb = 0.34 V → Ic = Is_eff · exp(0.34/Vt) ≈ 5×10⁻¹⁵ × 5×10⁵ = 2.5×10⁻⁹ A. So NPN delivers ~nA, not the µA needed.

The body-charging input loop is:
- Iii_M1 + Iii_M2 → into body
- BJT base current Ib + body diode currents → leaving body

Self-consistency: at our Vb=0.34, Iii_M1+Iii_M2 ≈ 1×10⁻¹⁵ (per A.1.d
trace). That's just enough to balance Ib ≈ Ic/Bf = 2.5e-9/10000 = 2.5e-13.
So the loop is balanced at low Vb. No positive feedback can form because
Iii doesn't grow with Vb (depends on Vds, not Vb directly).

**Why does Sebas's SPICE see Id = 2e-5 here?** Possibilities we can't rule out:

(a) His SPICE finds a *different* fixed point we don't (we've checked
    naive multi-root via Vb_init sweep — disproven), maybe via a
    transient-residue effect that DC sweep ignores.

(b) His foundry-supplied BSIM4 has additional impact-ion physics
    (length-binning via lalpha0, lbeta0; Vbs feedback into alpha0)
    we don't model.

(c) The "complementary bipolar" he refers to is something we still
    don't understand from the schematic. The schematic itself only has
    M1, M2, Q1 (NPN with emitter=GND), and a 1 fF body cap — no
    behavioural sources.

(d) Our arclength solver's no_grad path-trace is finding the wrong branch
    when the I-V is multi-valued in (Vb, Vd) — i.e. there IS a high-Vb
    solution at this Vd that Sebas's transient SPICE finds, and our
    DC arclength misses.

## Specific questions

1. **Which of (a)/(b)/(c)/(d) above is most likely?** Rank with one-sentence
   reason per option. Read `compute_iimpact` in `nsram_bsim4_port.zip`
   and tell us if you spot anything BSIM4-specific we're missing
   (length-binning of alpha0 maybe wrong sign?).

2. **The lalpha0 puzzle.** The M2 card has `alpha0 = 7.83756e-5` and
   `lalpha0 = -9.5e-7`. With `lalpha0/Leff` for Leff=1.78 µm, the bin
   correction is `−0.534` — **larger in magnitude than alpha0 itself,
   which would flip alpha0 negative**. But our code (`temp.py`) doesn't
   apply lalpha0 binning. **Should we?** And if we do, with the
   manufacturer's value lalpha0=-9.5e-7, does that produce a positive
   alpha0_eff or a negative one? The Berkeley v4.8.3 source treats this
   as `alpha0_eff = alpha0 + lalpha0/Leff` — what units does Berkeley
   expect for lalpha0?

3. **Body-charging avalanche bistability.** In Sebas's HSPICE deck,
   could there be a high-Vb stable solution (Vb~0.7-0.9 V) that our
   homotopy + arclength can't find because we sweep Vd from below
   instead of above the firing threshold? Concretely: can you propose
   a *direct* test using our existing solver (e.g., warm-start from
   Vb=0.85 then Vd-sweep backward) that would expose such a solution
   if it exists?

4. **The "complementary bipolar current" from Sebas's email.** Our
   subagent A.1.i decoded the LTSpice schematic and found NO
   behavioural source — only the standard NPN with `is=5e-9 bf=10000`.
   But Sebas wrote (Apr-17): *"I'm only including a complementary
   bipolar current to capture the full swing of the firing mechanism"*.
   What other interpretation of that phrase fits a vanilla LTSpice
   schematic? Is "complementary" referring to *the parasitic Q1 itself*
   relative to the M1+M2 channel current (i.e., a parallel current path
   at the drain — that fits emitter=GND), or could it be a model-card
   parameter / SUBCKT we're missing?

5. **Concrete next test to run tonight.** Given everything above, what
   is the SINGLE highest-information experiment we can do in <1h on
   Ikaros that distinguishes (a)/(b)/(c)/(d)? If multiple, prioritise.

## Files in the packet

- `prompt.md` (this)
- `fit_vs_meas.png` — current best, median 2.03 / p90 5.03
- `summary.json` — current run summary
- `M1_130DNWFB.txt`, `M2_130bulkNSRAM.txt` — Sebas's two foundry cards
- `parasiticBJT.txt` — NPN model card from his lab
- `2tnsram_simple.asc` — LTSpice schematic
- `2Tcell_BSIM_param_DC.csv` — Sebas's per-bias fitted parameters
- `email_history.md` — Eric ↔ Sebas ↔ Mario ↔ Robert thread
- `A1*_*.md` — all 9 prior diagnostic reports (so you don't redo work)
- `nsram_bsim4_port.zip` — full PyTorch port: compute_dc, leak (iimpact, gidl, igb), bjt (Gummel-Poon), diode, model_card parser, the 2T cell wrapper (post emitter=GND fix), the pseudo-arclength solver
- `validation_scripts.zip` — z91d, z91e, z91f, z91g

## What we'd like back

Short, ranked, actionable. Single page is fine. **Prefer concrete code
patches or one-liner test commands over prose.**

If you're Gemini Pro: please attempt the calculation for question 2
(unit analysis of lalpha0). If you're GPT-5: please cite the Berkeley
BSIM4 v4.8.3 source line for the lalpha0 binning (we don't have it
checked in here).
