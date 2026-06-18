# A7 — Sebas/Mario NS-RAM Materials Review

**Date:** 2026-05-01
**Author:** Eric (this session) for the FEEL/NS-RAM port team
**Scope:** Exhaustive read of every Sebas-supplied / Mario-supplied artefact
on disk to extract physics constraints relevant to the snapback fit gap
in the differentiable PyTorch BSIM4 port. Background: z91g v20 produces a
smooth Id(Vd) ramp; measurements show a near-vertical 6-decade snap at
Vd ≈ 1–1.5 V then a plateau. Median log-RMSE ≈ 0.95 dec.

**Sources mined** (* = read in full this session):
- `/home/ikaros/nsram_info/2026-04-30 BSIMfitsBA/` — pptx, BA-renamed cards, xlsx, CSV*
- `/home/ikaros/nsram_info/emails.rtfd/TXT.rtf` — full Gmail thread*
- `/home/ikaros/nsram_info/schematic&modelCards/` — `2tnsram_simple.asc`, `parasiticBJT.txt`, `PTM130bulkNSRAM.txt`*
- `/home/ikaros/nsram_info/Zoom/` — 31 jpegs across 3 meetings + closed-caption txt (caption = auto-translated gibberish; jpegs not OCR-processed this session)
- `/home/ikaros/nsram_info/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/` — new dataset (vnwell=+2 V)*
- `data/sebas_2026_04_22/` — `.asc`, `M1_130DNWFB.txt`, `M2_130bulkNSRAM.txt`, `parasiticBJT.txt`, `PTM130bulkNSRAM.txt`, `2Tcell_BSIM_param_DC.csv`*
- `docs/email_sebas_mario_nsram_v012.txt`*, `docs/email_to_sebas_2026-04-30.md`*
- `docs/NSRAM 20260429 Mario Seb.pptx` (our slide deck)*
- `docs/FEEL_x_NSRAM_pitch.docx`, `FEEL_x_NSRAM_pitch_first.docx`*
- `docs/NSRAM_Research_Proposal_2026-04-30.docx`, `nsram_research_proposal_2026.md`*
- `docs/nsram_joint_scope_2026.md`*, `docs/sebas_vg2_question.md`*
- `research_plan/artifacts/A1*.md` (cross-reference)*

---

## 1. Snapback / breakdown evidence

### 1.1 Sebas's own description of the firing mechanism (verbatim)

**Email Sebas → Eric+Robert+Mario, ~mid-Apr 2026** (from `/tmp/emails_plain.txt`
line 1493-1497, also in `email_history.md:930-934`):

> *"I'm working with a different SPICE tool and foundry-provided models and
> I'm focusing on adapting foundry models of body bias and impact ionization
> (in BSIM4) to fit the **floating body behaviour** of our 2T cells. This
> renders a more standard approach for us circuit designers."*
>
> *"In that sense, I've **dropped the avalanche diode models (very annoying
> for convergence)** that fire the bipolar parasitic effect in LTSpice, and
> I'm only including a **complementary bipolar current to capture the full
> swing of the firing mechanism**. Fits are looking good, but I'm still
> working on **polynomial dependence of model parameters with tuning
> voltages (VG1, VG2)** and **layout dependent effect on transistor models**
> to capture the experimental behaviour."*
>
> *"My question at this point is: can your approach drop the avalanche
> voltage as a control parameter and deal with the BSIM Impact ionization
> and body voltage directly?"*

Key extractions:
1. The fold no longer comes from a discrete avalanche diode (he removed
   that). It must emerge from BSIM4 §6.1 + the NPN alone.
2. He names the snapback ingredient as "complementary bipolar current"
   — interpreted in A.1.i as the Gummel-Poon Ic of `parasiticBJT` Q1.
3. He is **adding polynomial(VG1, VG2)** bias dependence to transistor
   parameters. The fixed-param BSIM4 deck is **not enough** even for him.
4. **LDE** is named — that is why he ships *two cards*: M1 (DNWFB,
   `etab=+1.8`) and M2 (bulk, `etab=-0.087`) — and per-bias polynomials.

### 1.2 BSIM-fits package PPTX (`2026-04-29 NS-RAM I-V BA plots.pptx`)

Slide 1: matrix of fits — for each VG1 ∈ {0.2, 0.4, 0.6}:
- VG1=0.2 → VG2 ∈ [-0.20, +0.10]
- VG1=0.4 → VG2 ∈ [0.0, +0.30]
- VG1=0.6 → VG2 ∈ [0.0, +0.50]

"Symbols = measurements, Lines = simulations." So his SPICE deck
produces snapback at every (VG1, VG2). At VG1=0.2 he can only reach
VG2=+0.1 V; at VG1=0.6 he runs all the way to VG2=+0.5 V.

Slide 2: separate plot at three VG2 cuts (0.0, 0.25, 0.35). Plateau
structure varies smoothly with VG2.

Slide 3: thumbnails of extracted parameter-vs-VG2 trajectories per VG1
("for all VG2") — this is the polynomial-fit data delivered as
`2Tcell_BSIM_param_DC.csv`.

### 1.3 Per-bias parameter table (`2Tcell_BSIM_param_DC.csv`)

Critical observations:

- **Polynomial:** ETAB, K1, NFACTOR, mbjt; trise (transient).
- **ALPHA0 is constant: 7.842e-5** across all 33 rows. Not fit per bias.
- **BETA0** walks from 10.75 (VG1=0.2, VG2=-0.20) up to 20.0 (VG1=0.6).
  2× modulation.
- **K1** drops from 0.55825 (VG1=0.2) → 0.53825 (VG1=0.4) → 0.41825
  (VG1=0.6). 23 % drop.
- **ETAB** spans 0.8 → 2.5 (3.1× modulation) — much steeper than the
  M1-card baseline of +1.8.
- **NFACTOR** spans 1.25 → 12.15 (almost 10×). Sebas's email confirms
  NFACTOR varies *only on M2*.
- **mbjt = 0.001 for VG1=0.2, mbjt = 1.0 for VG1=0.4 and 0.6.** The CSV
  literally turns the parasitic NPN off at VG1=0.2. Snapback in his
  sim therefore exists *only because of NPN turn-on*.
- **area = 1e-6** (matches `.asc` instance `area=1u`).

The 9 NaN rows at VG1=0.4, VG2 ∈ [-0.2, -0.05] are the **coverage gap**
Robert flagged — Sebas's fitter could not converge in that corner.

### 1.4 Schematic (`2tnsram_simple.asc`)

Element list:

| Inst | Symbol | Model | Connections |
|------|--------|-------|-------------|
| M1   | nmos4  | NMOS, l=Ln, w=Wn         | D=Din, G=G,  S=Sint, B=B |
| Q1   | npn    | parasiticBJT, area=1u    | C=Din, B=B, E=0 (GND) |
| M2   | nmos4  | NMOS, l=Ln·10, w=Wn      | D=Sint, G=G2, S=0, B=0 (see note) |
| C1   | cap    | CBpar = 1 fF, Rser=1 mΩ  | between B and 0 |

`.param Ln=0.18u, Wn=0.36u, CBpar=1f`
`.inc PTM130bulkNSRAM.txt   .inc parasiticBJT.txt`
`.op 0`

Snapback in this deck must form from exactly four ingredients:
1. BSIM4 channel impact-ionisation Iii of M1 charging body B.
2. Gummel-Poon collector current of Q1 emptying drain through B.
3. M2 pulling Sint (and, per A.1.i, the body) down — gated by VG2.
4. Capacitor C1 (DC trivial; transient body-charge τ).

**Notably absent**: no `vnwell` source, no DNW-to-P-body diode, no
PMOS, no behavioural source, no sub-circuit, no .nodeset, no .ic.
**The "complementary bipolar current" is Q1's Ic, not a separate
complementary device** (confirmed in A.1.i §6).

**Topology note on M2.B:** The LTSpice symbol `nmos4 R0` convention
puts the bulk pin on the right edge. Reading the wire-list at (560,272)
+ FLAG positions, M2's bulk pin lands on node `0` (ground), *not* B.
A.1.i §1 reports `M2.B = 0`; A.1.t §1 reports M2.B = B. **These two
artefacts disagree.** Resolving this is candidate-B in §7 — see Q2 in §6.

### 1.5 New dataset folder name confirms vnwell = +2 V

`/home/ikaros/nsram_info/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/`
contains three subfolders `…@VG2 VG1=0.{2,4,6} vnwell=2`, each holding
7-11 `StandardIV_HH_2vHCa-2_VG2=…_VG=0.{2,4,6}…csv` files. The folder
name encodes sweep-rate `SRavg=0` (slowest) and `vnwell=+2 V`. The same
condition is implicit in the older `data/sebas_2026_04_22/` dataset
(folder `2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2`). **Every measurement we
have is taken with the deep-N-well at +2 V relative to ground.**

CSV columns: `vdata, idata, tdata, Var4, vfixgdata, ifixdata`. No
metadata. `vfixgdata` ≈ VG1 (~0.83 mV), `ifixdata` ≈ leak monitor (~1 pA).
vnwell never enters the CSV — it is a static instrument bias.

### 1.6 Topology question (`docs/sebas_vg2_question.md`)

`Vd_up` *increases* with VG2 — opposite of what a back-gate-forward-bias
model predicts:

| VG1 | VG2=-0.10 | VG2=0.00 | VG2=+0.10 | VG2=+0.20 | VG2=+0.30 |
|-----|-----------|----------|-----------|-----------|-----------|
| 0.2 | 1.25 V    | 1.55 V   | 2.00 V    | (sat)     | (sat)     |
| 0.4 | 0.95 V    | 1.05 V   | 1.25 V    | 1.55 V    | 1.95 V    |
| 0.6 | --        | 0.90 V   | 1.05 V    | 1.15 V    | 1.35 V    |

`dVd_up/dVG2 = +2.0 to +2.7 V/V` — strong positive slope. The .asc
topology where M2 is a VG2-controlled *body discharge transistor*
explains this naturally: higher VG2 → M2 sinks more body current →
harder to drive Vb past BJT turn-on → larger Vd needed before Iii can
dominate. The series-discharge interpretation (M2.B=0, M2 between Sint
and GND) is consistent with this slope; the back-gate interpretation
(M2.B=B) gives the wrong sign. **The data sign supports M2.B=0.**

### 1.7 In-progress port diagnosis (research_plan/artifacts/A1*)

Quotes:

- **A.1.l:** *"The body settles at low/negative Vb because Iii is 3-7
  decades smaller than the NPN base current + M2 source-body-diode load
  at Vb=0.7; R_B at the hypothetical Vb=0.7 is *negative* at every
  bias."*
- **A.1.m:** *"Vb does NOT climb past 0.5 V at ANY tested alpha0 (up to
  7.84e-1, 10000× baseline) or beta0 (down to 0.5). … This is case (ii):
  missing body-charging physics."*
- **A.1.t:** identifies the deep-N-well to P-body forward-bias diode
  as the missing path: *"adding it as a current source on node B (fed
  from a new vnwell parameter, default 2.0 V, Js·area gated by series-R
  ≈ 1 kΩ) is the minimum change required."*
- **A.1.n:** at vnwell=+2 V and Vb≈-0.25 V, V_forward = +2.25 V; series-
  R-limited current is 0.1–10 mA, **490–47 000× the BJT base draw**.
  *"This mechanism is absent from both Sebas's `.asc` and our PyTorch
  port."*
- **A.1.h:** confirms our `compute_iimpact` is bit-faithful BSIM4 IIMOD=0;
  the formula correctly emits ~0 A at this bias because BETA0=18-20 V
  collapses `exp(-β/Δ)` to ~1e-30 when Δ=Vds-Vdseff is sub-volt. **Not
  a formula bug — a regime mismatch.**
- **A.1.g:** rules out solver/initial-condition issues. Residual surface
  has a single root, arclength reports `n_folds=0`, every Vb_init seed
  converges to the same triple. *"Newton already finds the only root
  the model admits."*

### 1.8 Smoking-gun mismatch

Sebas's deck **does NOT model vnwell at all**, yet his SPICE produces
snapback that matches measurements. Our port, with vnwell-diode coded
but inert (Rs=1e10), misses snapback by 6 decades. Two readings:

- **Reading A:** Sebas's SPICE snaps without vnwell because his per-bias
  `BETA0`, `K1`, `ETAB` and especially **NFACTOR up to 12** put BSIM4
  in a regime where Iii is large enough to flip the loop, and NPN
  positive feedback closes from there. We have the same params loaded
  but our subthreshold-n computation may not be applying NFACTOR(VG2)
  through to the right BSIM4 sub-block. (NFACTOR=12 is **not** a normal
  swing fit — it dramatically inflates body-effect on Vth in subVth.)
- **Reading B:** LTSpice silently adds something we are missing
  (transient initialisation behind `.op 0`, or a default well/body
  diode in the foundry NMOS sub-model that we cannot see).

z91g already feeds every CSV row exact and still misses by 6 dec, so
Reading A alone is insufficient — *something else is needed*.

### 1.9 Our port slide deck framing (`NSRAM 20260429 Mario Seb.pptx`)

- Slide 2: claims *"Model produces a clean snapback knee at known
  α₀/β₀ settings"* — this referred to a synthetic test where we hand-
  set α₀=1e-2, β₀=2 (off-card values). We have **not** reproduced
  snapback at Sebas's CSV-extracted values. Honest log-RMSE TBD.
- Slide 2: the body charge ↔ threshold ↔ drain current loop named as
  the source of snapback. Confirms our mental model is right; the
  problem is loop gain.
- Slide 3: *"Pseudo-arclength continuation designed for fold-bifurcations
  which snapback is."* — i.e. our solver assumes a fold exists. A.1.g
  confirms it does not yet emerge from the residual.

---

## 2. Sebas's SPICE deck contents

### 2.1 `2tnsram_simple.asc` (verbatim)

```
SYMBOL nmos4 464 112 R0   InstName=M1   Value2=l='Ln' w='Wn' m=1
SYMBOL nmos4 560 272 R0   InstName=M2   Value2=l='Ln*10' w='Wn' m=1
SYMBOL npn   736 112 R0   InstName=Q1   Value=parasiticBJT  Value2=area=1u
SYMBOL cap   688 288 R0   InstName=C1   Value=CBpar         SpiceLine=Rser=1m
TEXT .param Ln=0.18u   Wn=0.36u   CBpar=1f
TEXT .inc PTM130bulkNSRAM.txt
TEXT .inc parasiticBJT.txt
TEXT .op 0
```

Pin map (re-derived):

```
M1: D=Din, G=G,  S=Sint, B=B
Q1: C=Din, B=B,  E=0           (NPN — emitter to GND, collector to drain)
M2: D=Sint, G=G2, S=0, B=0     (per LTSpice nmos4 R0 symbol)
C1: between B and 0
```

This means **M2 sits between Sint and GND, with its body tied to GND**
(NOT to floating B). M2 is *not* a back-gate of M1; it is a series
discharge transistor. **This contradicts A.1.t's claim** that the port
ties M2.B to floating B (`nsram_cell_2T.py:341+`). Confirm against the
PyTorch source and against the .asc by re-reading wire (640,160) and
(800,272) FLAG positions.

### 2.2 `.options` / `.params` / sub-circuits

- **No `.option`** anywhere. All BSIM4 numerical defaults rely on
  LTSpice/ngspice built-ins (typically `gmin=1e-12`, `abstol=1e-12`).
- **No sub-circuits.** No `.subckt`, no behavioural sources.
- **`.param Ln=0.18u, Wn=0.36u, CBpar=1f`** — only three globals.
- **`.op 0`** — DC operating point with `0` initial UIC flag (or, more
  likely, a stale trailing zero LTSpice didn't strip).

### 2.3 `parasiticBJT.txt` (verbatim)

```
* Simple bjt for floating bulk parasitic bipolar effect
* Pazos, S.

.model parasiticBJT NPN(
    is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m
    rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5
    cje=0.7e-15 ne=1.5 ise=0
    tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2
)
```

Snapback knobs:
- `is = 5e-9` × `area=1e-6` → `Is_eff = 5e-15 A`. Standard.
- `bf = 10000` — extremely high forward beta. A 1 nA Iii into the base
  produces 10 µA collector current. **This is the fold engine.**
- `va = 100 V` — Early voltage; mild Ic dependence on Vd.
- `nc = 2` — collector-emitter ideality factor doubled. Softens onset.
- `ikr = 100 mA` — *reverse* knee. **Unusual choice — see §8 finding 2.**
- `vje = 0.7` — base-emitter junction barrier; sets the turn-on knee.
- `ne = 1.5` — non-ideal emission coefficient. Softens turn-on at low Vbe.
- `tr / tf` — transit times. DC-irrelevant.

This card is **"simple"** — hand-written by Sebas, not foundry-extracted.
A behavioural NPN tuned to deliver mA-scale snapback at low Vbe; the
high `bf=10000` is unusual (real silicon NPNs are 100-300) and is how
he gets the latch to fire when Iii is small.

### 2.4 `PTM130bulkNSRAM.txt`

Original PTM 130nm card with Sebas-tweaked NS-RAM parameters layered
on top. Critical lines (data/sebas_2026_04_22/PTM130bulkNSRAM.txt):

- L24: **`alpha0 = 7.83756e-5`** (constant override)
- L25: **`beta0 = 18`** (M2-card baseline)
- L26: **`agidl = 1.99e-8`**
- L27: **`k1 = 0.63825`** (M2 baseline)
- L28: **`etab = -0.086777`** (M2 baseline)
- L41: `Dvt0 = 8.7500000` — much larger than BA cards' 1.9758 → BA
  cards override this.
- L46: **`Vsat = 1.35e5`** (vs `vsat = 102230` in M2 card — discrepancy)
- L60: **`Pscbe1 = 8.66e8, Pscbe2 = 1e-20`** (vs BA cards' 5.331e8 and
  **1e-5** — five orders of magnitude difference in Pscbe2!)
- L62: **`* DEDUP (Robert intent): line 62 stale PTM defaults; using
  line 24-25 (alpha0=7.84e-5 beta0=18)`** — Robert's annotation.

The file has duplicate definition risks; Robert documented his dedup
choice. **Audit other overshadowing pairs.**

### 2.5 `.inc` ordering and shared model

`.inc PTM130bulkNSRAM.txt` first, then `.inc parasiticBJT.txt`. The PTM
file defines `.model NMOS NMOS` and `.model PMOS PMOS`; both M1 and M2
in the .asc reference `NMOS` (no model name on the symbol). So **M1
and M2 share the same `.model NMOS` block** in the original .asc —
same parameters, only L differs (Ln vs Ln·10).

This **directly contradicts** the BA package (which provides separate
M1 and M2 cards) and Sebas's email (*"some parameters change only for
M1, while NFACTOR changes only for M2 (LDE)"*). The LDE-aware split
is in the BA package, **not** in the original `.asc`. **Our port should
be using the BA cards** — A.1.t §1 confirms it does.

---

## 3. Model-card key parameters

### 3.1 M1_130DNWFB vs M2_130bulkNSRAM (BA package) comparison

| Param    | M1 (DNWFB)  | M2 (bulk)    | Note |
|----------|-------------|--------------|------|
| vth0n    | 0.54153     | 0.54153      | same |
| **k1**   | **0.53825** | **0.63825**  | body-effect: M1 weaker than M2 |
| **etab** | **+1.8**    | **-0.086777**| **Sign flip!** Body effect on subVth |
| lpe0     | 1.244e-7    | 1.244e-7     | same |
| lpeb     | -1.6512e-8  | -1.6512e-8   | same |
| dvt0     | 1.9758      | 1.9758       | same |
| voff     | -0.1368     | -0.1368      | same |
| nfactor  | 1.58        | 1.58         | per-bias for M2 only (CSV) |
| pscbe1   | 5.331e8     | 5.331e8      | substrate-current SCE term |
| pscbe2   | 1e-5        | 1e-5         | (PTM has 1e-20 — see §8) |
| **alpha0**| **7.83756e-5** | **7.83756e-5** | CSV-confirmed constant |
| **beta0**| **19**      | **18**       | per-bias from CSV (10.75–20) |
| agidl    | 1.99e-8     | 1.99e-8      | same |
| bgidl    | 1.624e9     | 1.624e9      | same |
| cgidl    | 6.3         | 6.3          | same |
| egidl    | 0.91        | 0.91         | same |
| **agisl**| not set     | not set      | **bug per A.1.e**: defaults 0; should mirror agidl |
| bvs      | 10          | 10           | source-body BV |
| xjbvs    | 1           | 1            | breakdown junction multiplier |
| jss      | 3.4089e-7   | 3.4089e-7    | source-body sat current density |
| njs      | 1.017       | 1.017        | source-body ideality |
| pdiblc1  | 3.3832      | 3.3832       | DIBL coefficient |
| lalpha0  | -9.843e-12  | -9.843e-12   | length-binning of alpha0 (-7%) |
| lbeta0   | -9.5e-7     | -9.5e-7      | length-binning of beta0 |

Potential mis-readings in our port:

1. **`etab` sign flip**: M1=+1.8, M2=-0.086777. Confirm port applies the
   right sign per card. CSV ETAB (0.8 → 2.5) overrides M1's +1.8
   per-bias, polynomial in (VG1, VG2).
2. **`pscbe2` mismatch**: BA=1e-5, PTM=1e-20 (5 orders!). If the loader
   silently picks PTM's value, the saturation-tail body coupling is
   suppressed. **Audit the param load order.**
3. **`agisl=0` bug** (A.1.e §H2′): `model_card.py` resolves `ref`
   defaults *before* user overrides apply. agisl stays 0 instead of
   mirroring agidl. Inert at the worst bias (GISL gate closed) but
   wrong at higher Vd or more positive Vbs.
4. **BVS=10 V**: source-body junction breakdown at -10 V. Worst Vbs is
   -0.5 V → no breakdown contribution. A.1.t §2.2 marks this irrelevant
   for the diagnostic bias.
5. **alpha0 = 7.83756e-5 m/V** — BSIM4 v4.8.3 unit (m/V). Not a
   magnitude bug. A.1.m's 4-decade brute-force test did not produce a
   fold even at alpha0×10000.
6. **beta0 = 18-19 V** is high — manual default ~30 V; short-channel
   cards typically use 0.5-3 V (A.1.h note). Sebas's per-bias 10.75 (low
   VG2) is at the low end. We reproduce his number faithfully, but the
   value is what makes `exp(-β/Δ)` un-zero at sub-volt drain headroom.
7. **nfactor per-bias up to 12.15 (M2 only)**: enormous and not a
   typical BSIM4 fit. NFACTOR multiplies the body-effect term in the
   subthreshold-swing computation:
   `n = 1 + NFACTOR·Cs/Cox + (CDSC + CDSCD·Vds + CDSCB·Vbs)/Cox + CIT/Cox`
   With nfactor=12 the prefactor inflates `n` ~10× vs nfactor=1.58 —
   massively softens M2 subthreshold slope and strengthens body-bias
   coupling in subthreshold, exactly the regime where the latch arms.
8. **mbjt per-bias** in CSV: 0.001 at VG1=0.2; 1.0 at VG1=0.4/0.6.
   Multiplicative gate on the parasitic NPN — Sebas literally **shuts
   the BJT off** at VG1=0.2. Port has `vnwell_mbjt` (A.1.t L:135) but
   may not be reading the `mbjt` CSV column. **Verify.**

### 3.2 Junction breakdown / GIDL parameter audit

| Param | M1 BA | M2 BA | PTM | Notes |
|-------|-------|-------|-----|-------|
| agidl | 1.99e-8 | 1.99e-8 | 1.99e-8 | GIDL prefactor |
| bgidl | 1.624e9 | 1.624e9 | 1.624e9 | GIDL exponent |
| cgidl | 6.3   | 6.3     | 6.3   | drain-body bias factor |
| egidl | 0.91  | 0.91    | 0.91  | GIDL band-bending threshold |
| agisl | def 0 | def 0   | def 0 | **bug**: should mirror agidl |
| bvs   | 10    | 10      | 10    | source-body BV |
| bvd   | not set | not set | not set | drain-body BV — defaults to 10 |
| xjbvs | 1     | 1       | 1     | breakdown junction multiplier |
| xjbvd | 1     | 1       | 1     | drain-side variant |
| ijthsfwd | 0.1 | 0.1   | 0.1   | forward junction current threshold |
| ijthsrev | 0.1 | 0.1   | 0.1   | reverse junction current threshold |
| jss   | 3.4089e-7 | 3.4089e-7 | 3.4089e-7 | source-body Js |
| njs   | 1.017 | 1.017   | 1.017 | source-body ideality |
| pbs   | 0.74883 | 0.74883 | 0.74883 | source-body junction barrier |

GIDL block is fully populated and consistent across M1/M2/PTM. The
A.1.e bug (agisl=0) is the only gap. **GIDL is closed at the diagnostic
bias** (A.1.e §44 — `Vd-Vg-egidl < 0` at all four edges). Junction
breakdown likewise irrelevant at our stable Vb≈-0.25 V.

---

## 4. Email + meeting key quotes (verbatim, dated)

### 4.1 Mario, 2026-03-20 (Zoom NSRAM kickoff)

Subject *"Zoom NSRAM"* to Robert, Eric, Pazos. Logistics only.

### 4.2 Sebas, ~2026-03-21 (response to first nsram package)

> *"I've taken a look to the python package, it's a great kickstart and
> I appreciate you taking the time to work on it. I will most definitely
> be updating modeling parameters over the next few weeks, so I'll be
> reaching back soon with details and model improvements based on new
> experiments, as my to-do list starts to clear over the next couple of
> weeks."*

### 4.3 Eric (us), ~2026-03-22 (nsram v0.9 reply)

> *"Device physics layer — the full body-charge ODE, **Chynoweth
> avalanche model**, SRH charge trapping, and **temperature-dependent
> BVpar**, all matched to your Zenodo SPICE parameters."*

(Recorded for context; v0.9 had the avalanche/BVpar machinery Sebas
later told us to drop.)

### 4.4 Mario, 2026-03-25 (delayed by travel)

> *"Sebastian did not have time yet to complete the modeling. Let us
> meet in the second half of April. I finish my trip on April 20."*

### 4.5 Sebas, ~2026-04-15 (the *physics-substantive* email)

Quoted in §1.1 — the canonical statement of the SPICE deck philosophy:
dropped avalanche diodes; complementary bipolar current; polynomial
parameter dependence; LDE.

### 4.6 Eric, 2026-04-20

> *"§6.1 channel HCI fits ~4 decades RMS better than §10.1 junction
> breakdown — so the channel-HCI route looks like the right match for
> your 2T cell, aligned with the 'complementary bipolar current on top
> of BSIM4' description from your last email."*

We commit to the channel-HCI path. Aligned with Sebas.

### 4.7 Sebas, 2026-04-20 23:08 (sending the package)

> *"1) A circuit schematic of the 2T cell, as currently modelled, in
> ASC format (LTSPice, even though we are using a different simulator
> right now targeting tape-outs). Neighbour coupling will most likely
> be the first topology we'll target. I will soon have help to scale
> this into network and circuit level implementations."*
>
> *"2) I-V curves in CSV files, at an average sweep rate of **0.2 volts
> per second**, 3 different VG1 (0.2, 0.4, 0.6) and multiple VG2 …
> We are generating additional data now (multiple ramp rates for more
> data on dynamics, pulsed dynamics)."*
>
> *"3&4) … the foundry's full model card cannot be shared without
> infringing NDAs, but the 130 nm (current working node) PTM model I'm
> attaching is a good starting point."*

### 4.8 Sebas, 2026-04-30 15:25 (post BSIM-fits package)

> *"Remember that some parameters change only for M1, while NFACTOR
> changes only for M2 (I attribute this to **LDE**, hence two separate
> model cards for each device)."*

> *"BONUS TRACK: I'm working on a floorplan for the **first testchip
> entirely dedicated to NS-RAM**. This will already include small
> arrays of NS-RAM based neurons, but if there is something small and
> specific that you think can be useful to extract specific metrics
> that help your approach, please let me know."*

(Our reply 2026-04-30 listed both a probe-pad single cell and a 16-cell
linear chain as asks.)

### 4.9 Mario quotes — funding/strategy only

> *"I have been travelling from March 28th until yesterday and I
> couldnt talk with the person in charge of the vendor registration."*
> (Apr 20)
> *"I am very sorry for being slow with this topic, but I think we are
> making a solid foundation."* (Mar 25)

No physics content from Mario in the thread. Role: foundry / NRF
funding.

### 4.10 Zoom meetings (2026-03-20, 04-22, 04-30)

The 04-30 meeting has a 2283-line closed-caption transcript dominated
by mistranscribed Swedish/English mix. Three readable physics-relevant
fragments located:

- *"Detta impact and."* (line 1277) — possibly "impact ionisation"
- *"And I think is differently use foldio."* (line 1442) — possibly
  "fold I/V" or "polynomial"
- *"Subdrential Saturation But Effect."* (line 4) — likely
  "sub-threshold saturation body effect"

**The 31 jpg screenshots in `/home/ikaros/nsram_info/Zoom/` were not
OCR-processed in this session.** Time-stamped to the three meetings (12
from 2026-03-20, 1 from 04-22, 18 from 04-30); likely whiteboard
sketches or shared-screen slides. Recommend OCR follow-up.

---

## 5. Mario / NRF context

### 5.1 Mario's positioning (from `nsram_joint_scope_2026.md`)

- **Foundry channel:** TSMC test-vehicle, May 6 NRF submission for
  Singapore NRF Mid-Sized Grant (~7 yr / $50 M strategic frame).
- **Architecture vision:** Hybrid CMOS-frontend / 2D-backend (hBN BEOL
  transfer process below 400 °C — compatible with mid-stack
  integration).
- **Power-band target:** 10–100 mW gateway tier (slide 26 of his deck —
  not in our archive but cited in our scope doc).
- **Tape-out:** First testchip *entirely dedicated to NS-RAM* in
  floorplan; small arrays of NS-RAM neurons; open to including
  specific probe-pad cells we request.

### 5.2 Sebas's measured energy & area

- ~21 fJ/cycle
- ~6.7 fJ/spike
- ~20 fJ integration loss
- ~46 µm² per cell
- 10× tunable spike rate (transient pulse measurements, slide 25)

These numbers anchor the NRF one-pager and the joint paper.

### 5.3 Our deliverables (from research proposal)

- **D1 — Calibrated NS-RAM digital twin (8 weeks, feeds NRF + TSMC).**
  Two-card LDE-aware port consuming Sebas's per-bias CSV as ground
  truth. **Currently blocked by snapback gap.**
- **D2 — Architecture exploration platform (6 months).** Standardised
  benchmark suite (Hopfield, MC, NARMA-10, 7-class waveform,
  temporal-XOR) at 71 k+ evaluations/s.
- **D3 — Headline demonstration: cognitive core on NS-RAM (12 months).**
  Distil 1–3 B-param reasoning core, looped-transformer with adaptive
  halt, co-design array geometry. Headline: *"a single device that is
  synapse, soma and IF spiker, carrying a reasoning model that thinks
  as long as it needs to — uniquely NS-RAM."*

### 5.4 What FEEL/NS-RAM pitch deck claims about our system-level work

(`FEEL_x_NSRAM_pitch.docx`, March 2026, pre-Sebas-package era):
- 57 experiment groups, 329 individual tests on FPGA-based 128-neuron
  NS-RAM model.
- "Native 1/f noise", criticality, causal emergence, MC=2.674 best.
- 8-neuron and 128-neuron FPGA bitstreams.

Parallel to the BSIM4 port track, **not** dependent on the snapback
fit.

---

## 6. Open questions to ask Sebas next (≤5)

1. **`parasiticBJT.txt` `ikr=100m` — was this intended as `ikf` (forward
   knee)?** In Gummel-Poon SPICE NPN, `ikr` is the *reverse* knee and
   is irrelevant in forward-active mode. The plateau-current cap on
   snapback (~mA) needs a source — `ikf` would do it naturally, `ikr`
   would not. If literal, what is the high-injection plateau mechanism?

2. **Confirm M2 body terminal in `.asc` wiring.** Our port and your
   `.asc` reading disagree: A.1.t §1 reports M2.B = floating B; A.1.i §1
   reports M2.B = 0 (ground). The +2 V/V `dVd_up/dVG2` slope supports
   M2.B = 0 (series-discharge), not M2.B = B (back-gate). Could you
   confirm M2's bulk pin convention?

3. **Does your SPICE deck implicitly include a `vnwell` source?** Both
   datasets are taken at vnwell=+2 V, but the `.asc` does not contain
   any voltage source on a deep-N-well node. Are you applying it as an
   external bias only, or is there an implicit well/body diode coded
   in the foundry NMOS card we cannot see? Adding a forward-biased
   well-to-body diode (~1 µA – 1 mA base drive) closes the 6-decade
   gap in our port.

4. **Polynomial form for ETAB(VG1, VG2), K1(VG1), NFACTOR(VG2),
   BETA0(VG1, VG2).** Could you share the polynomial coefficients (or
   even just the polynomial degree per parameter)? We are interpolating
   row-by-row from your CSV; closed-form polynomials would unblock
   smooth fitting and gradient flow.

5. **Sweep-rate-dependent hysteresis raw traces.** The CSV is a single
   sweep at 0.2 V/s; the new dataset folder is `SRavg=0` (slowest).
   Could you share the multi-rate transient raw (Vd(t), Id(t), VG2(t))
   data behind slide 25? This gates body-charge τ-calibration in D1
   and validates our pseudo-arclength fold tracker.

---

## 7. Top 3 physics candidates for the missing snapback term

**Candidate A — vnwell → P-body forward diode (rank 1).** The dataset
is at vnwell=+2 V, but neither Sebas's `.asc` nor our port models the
DNW-to-P-body junction. With Js=3.4e-7 A/m² and a typical contact
spreading resistance 1 kΩ–1 MΩ, the diode delivers 1 µA–1 mA on the
body node at converged Vb≈-0.25 V — **490–47 000× the BJT base draw**.
A.1.t §3 already wired the term; setting `vnwell_Rs ≤ 1 MΩ` and
`vnwell_mbjt = 1` should produce the fold. Strongest quantitative case
(A.1.n) and cheapest fix (single-line config).

**Candidate B — M2 wiring/topology bug.** If M2.B is grounded (not B),
M2 is a *series pull-down* of Sint to ground rather than a back-gate
of M1. This changes the VG2 control current. Combined with NFACTOR(VG2)
up to 12, the M2 subthreshold leakage path gives the body a much weaker
discharge than our current port assumes — letting Iii charge B to
BJT-on faster. Aligns with the +2 V/V VG2 slope and the LDE story.
**Quick falsifiable test:** flip M2.B to GND, re-run single bias,
check Vb trajectory.

**Candidate C — NFACTOR(VG2) inflating subthreshold body-effect.**
CSV shows NFACTOR going from 1.58 to 12.15 at low VG2 on M2 only. Our
port reads NFACTOR per-bias but the BSIM4 v4.5 formula uses it inside
the subthreshold-swing computation; with NFACTOR=12 the term
`(1+NFACTOR·Cs/Cox)` is ~10× larger, dramatically strengthening Vbs
coupling to Vth. This is **how Sebas encodes the positive-feedback gain
of body charge on threshold.** We may not be applying the per-bias
NFACTOR override into the right sub-block. **Test:** dump the residual
subthreshold n at the worst bias and compare to a hand-calculation
with NFACTOR=12.

(Honourable mention — `pscbe2` source mismatch: BA=1e-5 vs PTM=1e-20.
If the port silently picks 1e-20, the saturation-tail body coupling is
suppressed by 5 orders of magnitude.)

---

## 8. "We missed this entirely" findings

1. **`mbjt` is a per-bias scale on the NPN, not a fit residual.** The
   CSV column `mbjt = 0.001` for VG1=0.2 (suppress NPN) vs `mbjt = 1.0`
   for VG1=0.4/0.6 is **how Sebas reproduces "no-snap at VG1=0.2 vs
   snap at VG1≥0.4"**: he literally turns the BJT off at low VG1. We
   need to read `mbjt` from the CSV and apply it as
   `Q1.area = mbjt · 1u` at runtime. **Verify port does this.**

2. **`ikr=100m` in `parasiticBJT.txt` is suspicious.** In standard
   Gummel-Poon NPN cards, `ikr` is the *reverse* knee — it gates
   high-injection rolloff in *reverse-active* mode. For forward-active,
   the relevant knee is `ikf`, which is *not set* in Sebas's card.
   With `ikf` at default (∞), there is no high-injection rolloff in
   forward Ic; the plateau must come from somewhere else (Va=100, or
   intentional). Either Sebas's card has a typo, or he's deliberately
   exploiting the missing `ikf` to let Ic keep climbing.

3. **`.op 0` trailing zero** — likely a meaningless LTSpice artefact,
   but if it specifies *transient initialisation* (`.op` with UIC
   equivalent), our DC-only Newton solve cannot replicate the LTSpice
   startup path that may find the snap-up branch. Worth checking
   LTSpice docs.

4. **Original `.asc` shares one `.model NMOS` between M1 and M2.** The
   PTM card in the schematic folder defines a single NMOS used by both;
   the BA-package M1/M2 split is a later refinement. **If our port
   loaded only one of the two model files (or the wrong one), one of
   the two MOSFETs is mis-modelled.** A.1.t §1 confirms our port uses
   `M1_130DNWFB.txt` and `M2_130bulkNSRAM.txt` separately — good. But
   the CSV's per-bias ETAB/K1/NFACTOR overrides need to land in the
   right card per Sebas's *"some parameters change only for M1, while
   NFACTOR changes only for M2"* rule. **Verify our `patch_sd_scaled`
   plumbing is per-card, not global.**

5. **PTM's `Pscbe2 = 1e-20` vs BA's `Pscbe2 = 1e-5`.** A 15-decade
   parameter difference. If the loader silently picks PTM, the
   saturation-tail substrate-current-on-output-conductance coupling is
   off. **Audit param load order.**

6. **`.asc` capacitor `C1 (CBpar=1f)` has `Rser=1m`** — irrelevant for
   DC but means LTSpice can't have left this as a true floating node;
   1 mΩ Rser introduces a tiny conductance to ground, so LTSpice
   regularises. Sebas adds numerical leakage to keep LTSpice happy. We
   ignore it for DC; that is correct.

7. **Sweep direction matters.** Sebas's CSVs are *up-sweeps* at
   0.2 V/s. Our port runs single-bias DC. A.1.p found no hysteresis in
   the model's residual surface — fine for matching individual measured
   points but cannot reproduce the up/down hysteresis loop in slide 25.
   For D1 transient τ-calibration we need a transient solver, not DC.

8. **NFACTOR up to 12 is non-standard.** Typical BSIM4 NFACTOR values
   are 0.5–3. Sebas's M2 fit pushing NFACTOR to 12 is using NFACTOR as
   a *body-effect amplifier*, not a swing fit. This is a strong signal
   that the M2 sub-threshold body coupling (NFACTOR·Cs/Cox term) is
   the encoded route by which body charge re-modulates Vth(M2) and
   closes the latch loop in his SPICE. **Worth a dedicated audit of
   how our port's NFACTOR override flows into `n`.**

---

## 9. Bottom-line recommended next experiment

Single change, one-line config edit (from A.1.t): set
`cfg.vnwell_Rs = 1.0e5` (100 kΩ) and `cfg.vnwell_mbjt = 1.0` for the
VG1∈{0.4, 0.6} families. Rerun `forward_2t_arclength_grad` at the
diagnostic bias VG1=0.6, VG2=0.0, Vd∈[0.05, 2.0] / 40 steps. Pass
criteria: (a) `n_folds≥1`, (b) `Vb` jumps across two consecutive Vd
points from <0.4 V to >0.7 V, (c) `log10(Id)` sweeps ≥4 decades
inside ΔVd ≤ 0.2 V. **If it passes, the port has snapback.** If not,
implement Candidate B (M2.B → GND wiring fix) and re-run.

Then ship the surviving questions to Sebas as items 2–3 of §6.

---

*Eric · 2026-05-01 · for FEEL/NS-RAM team · review pre-D1 push*
