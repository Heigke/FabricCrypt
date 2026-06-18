# R-40 Deep Audit — Mario/Sebas Material Re-Scan

**Date:** 2026-05-14
**Trigger:** DC fit plateaus at ~4-5 dec for VG1=0.4/0.6 despite R-20+R-29+R-37+R-39
**Method:** Every Zoom/, schematic, model card, transcript, mail re-read; cfg flags audited

---

## 1. Findings table — by file

| File | NEW finding | In pyport? |
|---|---|---|
| `Zoom/mail.txt:351` (1 May 26) | Sebas's simulation had a "parasitic diode that wasn't working as expected (or wasn't working at all)" — **the very bug he caught and patched**. Now uses `pdiode area=5×4.4µm² between V_B and V_Nwell`. | `body_pdiode_to="off"` by default. **OFF.** |
| `Zoom/mail.txt:181-183` | Sebas IS using `Vnwell=2V`-biased deep-N-well in his SPICE deck; this is *external bias* applied at the package pin (no node in .asc). | pyport `vnwell=2V` knob exists but `use_well_diode=False`. **NEVER touched** by z346/z358. |
| `Zoom/mail.txt:221` (Sebas slide 21) | Image 21 shows the "Dynamic response with p-diode update" — i.e. fixing this single circuit element was important enough to send a separate email. | Not implemented as default. |
| `Zoom/schematic&modelCards/2tnsram_simple.asc` | M1 length `Ln=0.18u` (180 nm) — **schematic disagrees with model card** which has `lmin=lmax=1.3e-7`. M2 length `Ln*10 = 1.8 µm`. | pyport defaults Ln=180 nm ✓ but model card has 130 nm binning |
| `Zoom/2026-04-30 BSIMfitsBA/2Tcell_BSIM_param_DC.csv` | (re-confirm) `BETA0=20` at VG1=0.6 (NOT 19/18 from base card). `K1=0.41825` (NOT 0.638 base). `ETAB ∈ [0.8, 2.5]` (NOT base 1.8/-0.087). `NFACTOR(M2) ∈ [1.25, 12.2]` (NOT base 1.58). `mbjt=0.001` only at VG1=0.2 (i.e. **BJT essentially OFF** for VG1=0.2, fully ON for VG1=0.4/0.6). | Routed correctly via `make_overrides` for valid rows. |
| `Zoom/2026-04-30 BSIMfitsBA/2Tcell_BSIM_param_DC.csv` rows 10-13, 20-23 | **HIGH-VG1 LOW-VG2 ROWS ARE ALL NaN** (VG1=0.4 VG2∈[-0.2,-0.05]; VG1=0.6 VG2∈[-0.2,-0.05]). Sebas's fit *itself* could not converge there. We are comparing pyport against silicon at biases Sebas himself could not fit. | pyport `make_overrides` returns no overrides on NaN — falls back to **base card** ETAB=1.8, K1=0.638, BETA0=19. The pyport `4-5 dec` at high-VG1 is therefore an apples-to-oranges comparison. |
| `Zoom/schematic&modelCards/parasiticBJT.txt` | `nc=2` (B-C leakage emission), `vje=0.7`, `cje=0.7e-15`, `tf=25e-12`, `xtf=2`, `itf=0.03`, `vtf=7` | nc=2 ✓; junction caps + transit times **OFF** in DC port. Correct for DC. |
| `Zoom/data/sebas_2026_05_02/three_branch_params_extracted.json` | BETA0 has THREE qualitatively different regimes (red=10.75→14, blue=19 flat, black=20 flat) — branch-by-branch fits **structurally needed**. | poly fit `_eval_poly_d3` smooths across branches. **Polynomial cannot represent flat-step-flat.** |
| `Zoom/pdiode.txt` (level=1 SPICE diode) | `is=5.37e-7` is TOTAL (not per-area), `cj=7.33e-4 F/m²`, `bv=11` (no breakdown intended), perimeter card too | `body_pdiode_to="off"` default; even if on, `body_pdiode_Js` default is 1e-6 A/m² not Sebas's 24400. |
| `Zoom/2026-04-30 13.03.27 Zoom NSRAM/meeting_saved_closed_caption.txt` | Garbled Swedish-English ASR; no extractable physics statements beyond a brief "impact" mention | n/a |
| `Zoom/2026-04-29 NS-RAM I-V BA plots.pptx` | Plot labels only ("Thick=meas, Thin=sim", "VG2=0…0.5 step 0.05"). No equations. | n/a |
| `Slow I-Vs SRavg=0/...` | Three subdirs (VG1=0.2/0.4/0.6), one .csv per VG2 value. **Same data** as `data/sebas_2026_04_22/` 33-CSV set. Format: `vdata,idata,tdata,Var4,vfixgdata,ifixdata` (Var4 is the sweep rate marker). | Already loaded. |
| `M1_130DNWFB(M1).txt:50` vs `M2_130bulkNSRAM(M2).txt:63` | M1 base ETAB=**+1.8**, M2 base ETAB=**−0.087** — huge difference in subthreshold body-effect, and pyport's `M2_STATIC_OVERRIDES` correctly forces M2 to −0.087. | ✓ |

---

## 2. cfg flag audit — `NSRAMCell2TConfig`

| Flag | Default | Should Sebas's setup use? | Status |
|---|---|---|---|
| `use_iii` | True | YES | OK |
| `use_gidl` | True | YES (M1 card has agidl=1.99e-8) | OK |
| `use_bjt` | True | YES, with mbjt-scaled area | OK |
| `use_igb` | True | YES | OK |
| `use_diode` | True | YES (BSIM4 S/D junctions) | OK |
| **`use_well_diode`** | **False** | **YES** (Sebas vnwell=2V external bias) | **OFF — likely missing path** |
| `vnwell` | 2.0 V | 2.0 V | knob OK, not used |
| `vnwell_Rs` | 1e10 Ω | unknown | dormant |
| **`body_pdiode_to`** | **"off"** | **"vnwell"** (Sebas 1-May email is the fix) | **OFF — Sebas's headline fix not enabled** |
| `body_pdiode_Js` | 1e-6 A/m² | **2.44e4 A/m²** (per Sebas's `is`/area math) | inconsistent |
| `m2_body_gnd` | True | YES (LTSpice nmos4 unconnected body→GND) | OK |
| `m1_diode_scale` | 1.0 | 1.0 | OK |
| `quasi2d_body` | False | not in Sebas model | OK |
| `enable_tat` | False | not explicit in Sebas | OK to leave |
| `bjt_emitter_to_gnd` | False (LTSpice .asc) / True (ngspice deck) | True for ngspice comparison | active scripts (z346, z358) DO set =True ✓ |

---

## 3. Top 3 "obviously missing" items

### #1 — Parasitic N-well diode V_B → V_Nwell is OFF (`body_pdiode_to="off"`)
**Evidence:** mail.txt:351 (Sebas's 1-May "I found that my simulation had a parasitic diode that wasn't working as expected … I'm updating my schematic with an additional diode for clarity (pdiode 5×4.4 µm²) to reflect the capacitive response of the floating body … in principle sorts out our issues around dynamic behaviours"). SA3 image-21 explicit. Sebas's own simulation jumped to "good agreement" *only after* this diode was added. We have his card (`Zoom/pdiode.txt`). Default in pyport stays "off" with the comment "production fit has 1.00-decade median without it; turning it on is a forward action".  **But the production fit was on a CLEANED ngspice baseline that we have since found broken (PHYSICS_VERDICT track T4).** With the now-correct R-37 IIMOD strength (1000× stronger Iii after binunit fix) and no body-pdiode, V_B is over-pumped at high VG1 — exactly the z358 symptom (VG1=0.6 → 5.64 dec).

### #2 — vnwell=2V external bias never injects current (`use_well_diode=False`)
**Evidence:** mail.txt:181, slide labels "vnwell=2", SA3 image-18 ("VNwell≥2 V" defines legal op window). Sebas's data is **explicitly named** `Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0`. The +2 V deep-N-well bias is the EXTERNAL boundary condition for every measurement. Pyport disables this path with the comment "BSIM4 internal handles it" — but BSIM4 only has `Vbs = Vb−Vs`, not a separate well node. The well/body forward-bias path is genuinely missing, only re-introducable via `body_pdiode_to="vnwell"` with use_well_diode flag set. This couples directly to #1 (same physical junction).

### #3 — At VG1=0.4/0.6 VG2≤−0.05, fit params are NaN in CSV
**Evidence:** `2Tcell_BSIM_param_DC.csv` rows 10-13, 20-23. Sebas's fit *itself* did not converge there. pyport's `make_overrides` silently falls back to base card ETAB=1.8, K1=0.638, BETA0=19 — i.e. completely wrong parameters for those VG2 values. The R-39 z358 "per-VG1 5.64 dec @ VG1=0.6" residual includes **4 NaN-bias rows where we use base-card params** averaged in with **7 fit rows** where we use Sebas params. We are penalising pyport for a regime Sebas himself couldn't fit. **The 33-bias loss is contaminated by 8 mismatched-parameter rows.**

---

## 4. Single highest-impact recommendation

**Enable `body_pdiode_to="vnwell"` with `use_well_diode=True` and `body_pdiode_Js` per Sebas's pdiode card** (rescale total is=5.37e-7 / area=22e-12 = 2.44e4 A/m², or load pdiode.txt verbatim).

This is the SINGLE change Sebas himself flagged in his 1-May email as "in principle sorts out our issues" — and it is the only physics change documented in the mail thread *after* the BSIM4 §6.1 IIMOD path was nailed down. It is the *clamping* element that prevents the now-correct (post-R-37) Iii ≈ ngspice from over-pumping V_B at high VG1. Without it, the strong Iii has no reverse-path to discharge, and the body floats up to ~Vd, dominating the channel current.

**Secondary, but free**: subset the 33-bias loss to non-NaN CSV rows when computing dec-residuals (or impute Sebas's branch-flat values: ETAB=2.5, K1=0.418, BETA0=20 for VG1=0.6 rows). This alone should drop the cell-wide median by ~0.3-0.5 dec just by removing the bookkeeping artifact.

**Tertiary**: V_B ↔ V_G2 designed coupling capacitor (SA3 finding 2) — but this is dynamic, won't affect DC fit.

---

## 5. Verdict

There IS a single obvious missing item that could plausibly close the gap: the **N-well/P-body diode that Sebas explicitly flagged in his last email**. We have his SPICE card. We have the knob in pyport. It is defaulted OFF based on a now-invalidated baseline (PHYSICS_VERDICT track T4 found that ngspice baseline broken by HSPICE-expression silent-drop; R-37 made Iii 1000× stronger which removes the "1.00 dec without it" justification entirely).

The 4-5 dec high-VG1 plateau is consistent with: strong, correct Iii pumping V_B; no reverse path to discharge; V_B latches near V_D; M2 sees Vbs ≈ V_D-0 ≈ +2 V (way out of fit range); Ids of M2 explodes. Turning ON the well-pdiode provides the V_B → V_Nwell discharge path with a turn-on at V_B ≈ V_Nwell ≈ 2 V, naturally clamping the body before latch-up.
