# DA2 — Data Audit (NS-RAM)

**Date**: 2026-05-12
**Scope**: every file under `data/sebas_2026_04_22/`, `data/sebas_2026_05_02/`,
`data/nsram_zenodo/SimulationFiles/SPICE/`, and BOTH TCAD subtrees
(`FloatBulk_Tsub/`, `FloatBulk_Rsub/`).
**Method**: directory walk + grep over `scripts/`, `src/`.

Legend: **USED** = at least one fit/sim script loads the file by path or its
parameters by name. **PARTIAL** = file loaded but a sub-block ignored.
**UNUSED** = no script references it.

---

## 1. `data/sebas_2026_04_22/` — DC silicon ground truth

### 1.1 BSIM4 model cards

| File | Type | Status | What's used | What's ignored |
|---|---|---|---|---|
| `M1_130DNWFB.txt` (120 lines) | BSIM4 card for M1 (DNWFB nmos) | **PARTIAL** | core long-channel params: `vth0, k1, k2, etab, nfactor, dvt0/1/2, dvt0w/1w/2w, eta0, dsub, voff*, lpe0/b, kt1/l, kt2, u0, ua/b/c/d, alpha0, beta0, w0, k3/b, cdsc*, cit, vsat, ags, b0, keta, pclm, pdiblc*, drout, pscbe*, pvag, delta` (see `bsim4_literal_port/bsim4_literal.py` — 47 keys) | **All output-stage params (≈80 names): `agidl/bgidl/cgidl/egidl, em, ef, noia/b/c, jss, jsws, mjs, cjs/sws/swgs, pbs/sws/swgs, xrcrg*, rbpb/d/s/db/sb/ps0..., rbody*, rbpbx/by/rbsbx/by/rbdbx/by..., gbmin, ute, kt1l, ua1/ub1/uc1, ud1, at, prt, njs, xtis, tpb*, tcj*, tvoff, tvfbsdoff, saref, sbref, wlod, ku0, kvsat, kvth0, llod*, lku0, lkvth0, wlod*, wku0, wkvth0, k2we, kvth0we, ku0we, web, wec, scref, kvsat`. ALL gate-tunnel currents (`igcmod=igbmod=0` so they would be zero anyway — but card values are loaded as data and dropped). All capacitance params (`cgso/cgdo/cgbo/cgsl/cgdl/clc/cle/dlc/dwc/vfbcv/noff/lnoff/voffcv/acde/moin`). All flicker-noise params (`fnoimod=1`, `noia/b/c, em, ef, lintnoi`).** |
| `M2_130bulkNSRAM.txt` (133 lines) | BSIM4 card for M2 + global `.param` block | **PARTIAL** | same 47 core params, PLUS the global `.param` line (toxn, vth0n, k1=0.63825, etab=-0.086777, beta0=18, vsatn, lpe0n, k3n, lintn, wintn, pvth0n) — these are the M1↔M2 deltas | Same 80 params as M1. Also the `rcjn/rcjp/rcjswn/...` capacitance-multiplier params (currently set to 1 in card; harmless) |
| `PTM130bulkNSRAM.txt` (129 lines) | base PTM card from which M1/M2 derive | **USED** as text loaded by z2510, z279 (no params extracted into pyport) | same 80 output-stage |
| `PTM130bulkNSRAM.original.txt` | byte-identical backup of above | **UNUSED** | — |
| `parasiticBJT.txt` (4 lines) | NPN Gummel-Poon one-liner: `is=5e-9, va=100, bf=10000, br=100, nc=2, ikr=100m, rc=0.1, vje=0.7, re=0.1, cjc=1e-15, fc=0.5, cje=0.7e-15, ne=1.5, ise=0, tr=20e-12, tf=25e-12, itf=0.03, vtf=7, xtf=2` | **PARTIAL** | `Bf` was acknowledged (z91b note); recent fits OVERRIDE Bf to 1e4 manually | **`va, br, nc, ikr, rc, vje, re, vje, ne, ise, itf, vtf, xtf, tr, tf, fc`** — all secondary Gummel-Poon shape & cap params NEVER passed to our NPN model. In particular `va=100` (Early voltage), `ne=1.5` (low-current ideality), `ikr=0.1` (high-injection knee) directly govern the BJT output conductance & soft turn-on we are currently approximating with a single ideal-diode + Bf. |
| `2Tcell_BSIM_param_DC.csv` (34 rows) | per-(VG1,VG2) extracted parameters: `VG1, VG2, trise, ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS, area` | **USED** by z91f, z91i poly, z273, z283 | `trise` column is loaded by z273 (transient sim) but never by the DC fit harness — explicit risk that we ignore time-domain anchor data Sebas already provides per-bias |

### 1.2 Raw silicon I-V CSVs

`2vHCa-2 I-Vs@VG2 VG1={0.2,0.4,0.6} vnwell=2/StandardIV_...csv` — 33 sweeps over (VG1,VG2). **USED** by z70/z91/z2510 fit harnesses (the canonical ground truth).

### 1.3 LTspice schematic

`2tnsram_simple.asc` — **UNUSED** as input to any sim script (we have re-implemented the topology in code rather than re-running it).

---

## 2. `data/sebas_2026_05_02/` — Newer drop

| File | Type | Status | Comment |
|---|---|---|---|
| `pdiode.txt` (10 lines) | LTspice `.model pdiode diode level=1` with: `is=5.37e-7, isw=1.37e-13, rsw=0.46, ns=1.085, nz=1.37e-13, imax=1e30, imelt=1e30, bvj=1e31, ik=97740, bv=11, ibv=97740, ikp=1.19e5, n=1.0535, rs=7.4e-8, cj=7.3e-4, cjsw=1.05e-10, vj=0.219, vjsw=0.652, m=0.241, fcs=0.5, mjsw=0.260, fc=0.5, xti=6.5, eg=1.11, tlev=1, tlevc=1` | **PARTIAL** | z112 reads ONLY `is, n` (Shockley two-param). z143/z275 use a hard-coded `body_pdiode_Js` and `_n` cfg pair. **17 of 19 model params ignored**: `isw, rsw, ns, nz, ik, bv, ibv, ikp, rs, cj, cjsw, vj, vjsw, m, mjsw, fc, fcs, xti, eg, tlev`. Most consequential: **`isw, ns, nz` (sidewall component)**, **`ik` (high-injection knee)**, **`rs=7.4e-8` (series R)**, **`bv=11, ibv=97740` (reverse breakdown)**, **`xti=6.5, eg=1.11` (temperature)**. The sidewall component is dimensionally what makes a junction diode `Js·A + Js,sw·P` — we are using `Js·A` only and dropping P (perimeter) entirely. |
| `image-2.png` | slide-deck schematic (titled "Experiment list 3-7") | **UNUSED** as data input | referenced only in 01_LOG.md prose. PNG-rastered analog of three_branch_params (see below). |
| `three_branch_params_extracted.json` | **manually transcribed** per-branch Sebas-fit tables: NFACTOR_M2(V_G2) red/black/blue, K1_M1(V_G1), ETAB_M1(V_G2) per branch, BETA0_M1(V_G2) per branch — **15-point grid on V_G2 ∈ [-0.2, 0.5] V** | **UNUSED — ZERO SCRIPT REFS** | This is the **highest-value unused asset**: Sebas's own structural decomposition of the parameter dependence into three V_G1 branches. Includes critical observations: (a) NFACTOR_M2 reaches **12.2 at V_G2=−0.2 V** (red branch) — well above our current BBO upper bound of 3.0, so our fit is structurally clamped; (b) BETA0 shows three regimes (red rising 10.75→14, blue flat at 19, black flat at 20) — **a single polynomial cannot fit this, branch decomposition is structurally necessary**. |

---

## 3. `data/nsram_zenodo/SimulationFiles/SPICE/dev/` — Mario's reference SPICE setup

| File | Type | Status |
|---|---|---|
| `AvalancheCircuit2_BulkMOSFET.asc` | LTspice schematic of the canonical M1+M2+Q1(avalBJT)+D1/D2(avalancheD) NSRAM cell. Has `.tran 0 {40*tr} {tr} {tr/10000}`, `.step param Vg list 0.25 0.45`, `Vd=1, Vgb=2.65, bvzener=2.45, Cbe=1p, tr=0.001` | **UNUSED** — schematic never re-rendered. **The `.tran` stimulus (Vpulse 0→3.5 V, `tr=0.001 s`, period 10·tr, 10 cycles) IS the transient regime our DC-only fit cannot capture.** |
| `BJTavalanche.txt` | `avalBJT NPN` model parameterized over `IsPar, BfPar, NfPar, VafPar, NePar, VarPar, CjPar, MjPar, BVpar, nbvPar` — these are the **avalanche-tunable** versions of parasiticBJT | **UNUSED** as data; structure replicated in code |
| `BJTparams.txt` | concrete defaults: `IsPar=1e-16, BfPar=50, VafPar=40, NfPar=0.9, NePar=1.5, VarPar=10, nbvPar=0` and a commented-out **bias-dependent `BVPar = '3.5-(1.5·Vg)'` (Tsinghua) vs `'1.6+(0.4/Vg)'` (TSMC)** — Mario's two-foundry recipe for the avalanche-onset V_G dependence | **UNUSED — high value**: gives a closed-form `BV(V_G)` to compare against our snapback model |
| `Davalanche.txt` | Zener-type `avalancheD` w/ `bv=0.9·BVPar, Rs=50, nbv=7, Ibv=1e-3, Ibvl=1e-3, Nbvl=0.15, Tbv1=-21.3e-6` | **UNUSED**: this is the canonical **zener-with-soft-knee** model for the body-diode breakdown — we use a plain Shockley `pdiode` |
| `subcircuit/NeuronSubCirc.asc/.asy` | sub-circuit wrapping the whole cell as a "neuron" with pins G/G2/D/B — proves Sebas/Mario package this as a single neural element | **UNUSED** |
| `subcircuit/SubC_SimpleTest.asc` | tiny test harness | **UNUSED** |
| `PTM130bulk_lite.txt` | Mario's reduced PTM card | **USED** (z2140) |

---

## 4. TCAD — `FloatBulk_Tsub/` vs `FloatBulk_Rsub/`

Both directories use the same `sde/nMOS` half-mesh, same IdVgs/IdVds/BV scripts; they diverge ONLY in `sdevice_des.cmd` (substrate boundary condition) and `sdevice15` parameter swept by Synopsys swb.

| Aspect | Tsub | Rsub |
|---|---|---|
| Bulk topology | `nmos_bsim3 mosnbulk(subs gb 0 0)` — an **M2 transistor body-leak path** | `Resistor_pset Rbulk(subs 0) { resistance = @Rsub@ }` — a **pure ohmic body-leak resistor** |
| swb sweep var | `VG2 ∈ {0.45, 0.5, 0.55, 0.6, 0.65}` | **`Rsub ∈ {0.1 Ω, 1e4 Ω, 5e5, 5e6, 5e7, 5e8, 5e9, 1e13 Ω}` (8 values)** |
| Sentaurus models dir | local `models/` w/ `models.scf` | (none — uses default models) |
| Status | partially audited by z299 | **NEVER REPLAYED — UNUSED** |

`Rsub` is the **canonical body-leak ladder**. We've fit to silicon at one implicit Rsub (whatever the silicon has) but never replayed the 8-value Rsub TCAD ladder. Critical because the Rsub→∞ corner is the "ideal float" limit and Rsub→0 is the grounded-bulk limit; the 4-decade sweep tells us which regime the silicon device lives in.

---

## 5. Summary

**Files audited**: 16 in `sebas_2026_04_22/`, 3 in `sebas_2026_05_02/`, 8 in `SPICE/dev/`, 33 in each TCAD tree (66 total ≈ 90 incl. duplicates) — **~93 unique files**.

**Unused / underused (concrete inventory)**:

1. `data/sebas_2026_05_02/three_branch_params_extracted.json` — **UNUSED**. Sebas's per-branch (V_G1=0.2/0.4/0.6) fit-parameter tables. **HIGHEST VALUE**.
2. `data/sebas_2026_05_02/pdiode.txt` — **PARTIAL**, 17 of 19 model params ignored (sidewall `isw/ns`, knee `ik`, breakdown `bv/ibv`, series `rs`).
3. `data/sebas_2026_04_22/parasiticBJT.txt` — **PARTIAL**, only `Bf` honoured. Missing `va, ne, ikr, ise` — these set the BJT soft-turn-on & Early.
4. `data/nsram_zenodo/SimulationFiles/SPICE/dev/BJTparams.txt` — **UNUSED**. Has Mario's `BVPar = 3.5 − 1.5·Vg` (Tsinghua) / `1.6 + 0.4/Vg` (TSMC) closed-form for V_G-dependent avalanche onset.
5. `data/nsram_zenodo/SimulationFiles/SPICE/dev/AvalancheCircuit2_BulkMOSFET.asc` — **UNUSED**. Canonical transient stimulus (Vpulse 0→3.5 V, `tr=0.001 s`, 10-cycle) we have never replicated.
6. `data/nsram_zenodo/SimulationFiles/SPICE/dev/Davalanche.txt` — **UNUSED**. Zener soft-knee body diode (Rs=50, nbv=7, Ibv=1e-3) — structurally different from our Shockley pdiode.
7. `data/nsram_zenodo/SimulationFiles/TCAD/FloatBulk_Rsub/` — **UNUSED ladder**. 8-value Rsub sweep over [0.1 Ω, 1e13 Ω], complementary to Tsub's VG2 sweep.
8. BSIM4 cards M1/M2 — **PARTIAL**: ~80 output-stage / capacitance / GIDL / flicker / temperature params loaded as text and dropped; pyport stops at Vgsteff and never computes Id, so all current-equation params (alpha0/beta0 used elsewhere by us manually, but the rest — pclm, pdiblc1/2/b, pscbe1/2, pvag, etc — are touched only by name in z91-fit boundaries, never propagated end-to-end into our drain-current).
9. CSV column `trise` — **UNUSED** by DC fit (only used by z273/z283 transient experiments).

**Highest-value finding**: `three_branch_params_extracted.json` plus the M1 card observation. The JSON explicitly states `NFACTOR_M2` hits **12.2 at V_G2=−0.2 V**, while our M3 fit BBO upper bound is **3.0** → we are clamping the fit at 1/4 of the physically observed range. Sebas also confirms **BETA0 requires branch decomposition** (flat-step-flat in V_G2), which our current single-polynomial(V_G1, V_G2) cannot represent — this matches the z2026-05-04 walk-back (1.39 dec, not 1.00 dec; ER_SPARSE wins MC, not MESH_4N).

**Most actionable next step (single experiment)**:

Add a `z240_three_branch_loader.py`:

1. Load `three_branch_params_extracted.json` → 4 tables.
2. Re-run pyport DC residual with NFACTOR_M2 bounds raised to **[1.0, 15.0]** (was [1, 3]).
3. Re-fit BETA0 as a **branch-indexed** parameter (red/blue/black driven by V_G1 ∈ {0.2, 0.4, 0.6}) instead of a single global.
4. Report dec improvement vs current 1.39-dec honest baseline. **Expected**: dec → 1.6–1.8 on the red branch where NFACTOR clamping is worst.

**Pitfalls**:

- `three_branch_params_extracted.json` was **manually transcribed from PNG rasters** (±5 % on values, ±0.01 V on V_G2 axis). The `_caveat` warns it's Sebas's SPICE-fit output, not silicon ground truth — use to constrain bounds & seed init, NOT as a fit target.
- Color→V_G1 mapping is **inferred** (red↔0.2, black↔0.6, blue↔0.4) — needs Sebas confirmation before publishing.
- Pyport stops at Vgsteff; broadening BSIM param coverage means porting the **rest of b4ld.c (§1337–end)** for drain-current — substantial work. Cheaper: do (3) using existing torch-based fit harness in z91 series which already computes a forward Id.
- `pdiode.txt` `is=5.37e-7 A` interpreted as **total** gives ~mA at Vb=0.5 V (unphysical for 22 µm² junction); z112 already flagged this. Adding `isw·P` makes the conflict worse, not better — likely need to interpret `is` as `A/m²` AND add the sidewall term.
- TCAD Rsub replay requires Sentaurus license (we don't have); however, the 8 Rsub values can be **mirrored in pyport** as a body-leak resistor sweep.
- Mario's `BVPar = 3.5−1.5·V_G` is the **Tsinghua foundry recipe**; Sebas's silicon is at imec (different process) so the closed form is structural-only — re-fit slope/intercept against our snapback data before using.
