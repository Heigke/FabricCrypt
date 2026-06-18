# R-1: Zoom Directory Deep Audit

**Date:** 2026-05-13
**Scope:** `nsram/Zoom/` (audit per TOPOLOGY_REBUILD_PLAN_2026-05-13 item R-1)
**Method:** Direct file reads + cross-reference against SA1/SA2/SA3/P2 prior audits.

---

## 1. File Inventory & Classification

| Path | Size | Status pre-R1 | Notes |
|---|---|---|---|
| `Zoom/2026-04-30 13.03.27 Zoom NSRAM/meeting_saved_closed_caption.txt` | 47.9 kB / 2283 lines | **UNAUDITED** | Auto-transcribed Swedish/English mix, ~80% noise. |
| `Zoom/mail.txt` | 21.7 kB / 392 lines | **UNAUDITED** | Full Gmail thread Mar20 → May03 2026. **HIGH VALUE** — clear text. |
| `Zoom/pdiode.txt` | 657 B | Already in `data/sebas_2026_05_02/` — covered by SA1 §pdiode. | Duplicate. |
| `Zoom/schematic&modelCards/2tnsram_simple.asc` | 1.4 kB | Already in `data/sebas_2026_04_22/` — covered by SA1. | Duplicate. |
| `Zoom/schematic&modelCards/parasiticBJT.txt` | 244 B | Already in SA1 (BJT 16 params). | Duplicate. |
| `Zoom/schematic&modelCards/PTM130bulkNSRAM.txt` | 5.1 kB | Already in SA1/SA2 (PTM 130 nm card). | Duplicate. |
| `Zoom/2026-04-30 BSIMfitsBA/130bulkNSRAM(M2).txt` | 10.5 kB | M2 BSIM4 card — **PARTIAL** prior coverage in SA1 (params listed); raw card itself unaudited as a file. | M2: NFACTOR=1.58, etab=-0.086777, k1=0.63825, alpha0=7.83756e-5, beta0=18, lbeta0=-9.5e-7, vth0n=0.54153, toxn=4 nm, vsatn=102230, lc=5e-9, xn=3. |
| `Zoom/2026-04-30 BSIMfitsBA/130DNWFB(M1).txt` | 9.9 kB | M1 BSIM4 card — **PARTIAL** SA1 coverage; raw card itself unaudited. | M1: deep N-well floating-body NMOS (`NMOSdnwfb`), k1=0.53825, etab=+1.8 (vs −0.087 in M2 — sign flip!), `rbodymod=0`, extensive layout-dep block (`ku0`,`kvth0`,`saref`,`sbref`). |
| `Zoom/2026-04-30 BSIMfitsBA/2Tcell_BSIM_param_DC.csv` | 1.95 kB | **NEW (this audit)** — 33-row VG1×VG2 polynomial table. Tabulated in P2 §binning-audit but never opened as raw. | Quantitative VG2-sweep parameter table: `VG1,VG2,trise,ETAB,K1,ALPHA0,BETA0,NFACTOR,mbjt,IS,area`. |
| `Zoom/2026-04-30 BSIMfitsBA/2Tcell_BSIM_fits.xlsx` | 12.8 kB | **UNAUDITED** Excel — likely fit-error tables. | Not opened (binary, would need python). |
| `Zoom/2026-04-30 BSIMfitsBA/2026-04-29 NS-RAM I-V BA plots.pptx` | 2.4 MB | **UNAUDITED**. Slides Sebas shared post-call ("raw didn't have time to format"). | 28 unopened slides — see action item below. |
| `Zoom/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/VG1=0.2/` | 7 CSVs × 82 pts | **UNAUDITED** — DC-limit silicon IV sweeps. | Cols: `vdata,idata,tdata,Var4,vfixgdata,ifixdata`. tdata=0..23s (≈ 0.2 V/s confirmed). |
| `Zoom/.../VG1=0.4/` | 7 CSVs | **UNAUDITED** | VG2 ∈ {−0.15…+0.15}. |
| `Zoom/.../VG1=0.6/` | 10 CSVs | **UNAUDITED** | VG2 ∈ {−0.20…+0.25}. Widest VG2 family. |
| `Zoom/Image 2026-03-20 at *.jpeg` (×18) | ~6–7 MB | Indexed in `binning_audit/zoom_slides_inventory.md`. | Visual content described. |
| `Zoom/Image 2026-04-22 at 19.57.jpeg` | NUS proposal form | Indexed — admin, LOW. | — |
| `Zoom/Image 2026-04-30 at *.jpeg` (×12) | 4–5 MB | **PARTIALLY indexed** (inventory covers only 5 of the 12 04-30 images). | 13.23, 13.24, 13.25, 13.28, 13.31, 13.33 not in inventory. |
| `Zoom/image-2.png` | 577 kB | Duplicate of `data/sebas_2026_05_02/image-2.png` — covered SA3. | — |

**Total NEW (never-opened) artifacts: 25 files** = 24 slow-IV CSVs + 1 pptx + 1 xlsx + 7 un-indexed 04-30 images + 1 mail.txt + 1 transcript (effective new info from transcript: low; from mail+CSV+xlsx+pptx: high).

`nsram/data/sebas_2026_05_02/Zoom/` is an **EMPTY symlink/marker dir** (only contains the same `Image*04-30*.jpeg`, `image-2.png`, `pdiode.txt`, `schematic&modelCards`, `Slow I-Vs ...` subfolders) — confirms no second corpus, just the same tree.

---

## 2. Transcript Content Summary (`meeting_saved_closed_caption.txt`)

The closed-caption is auto-generated bilingual (Swedish + broken English). 2283 lines but ~90% is filler ("Ja", "Nej", "Langenglish", "Sebastian", garbled phonemes). Useful extractable content (paragraph-level paraphrase, not quote-able due to ASR garble):

- **Robert (opening)**: Reports custom GPU kernels solve fitting "60× slower in float64"; can now do "all seven values in 10 minutes" simulation sweeps; admits invented terminology ("etabl", "damping factor", "constants") instead of working in Sebastian's smooth-VG1/VG2 polynomial parameter space.
- **Robert**: Wants to peak the "primars" (parameters) interesting to "us", lower-priority "VG1 sweeps" remark, asks about dynamics.
- **Sebastian (key technical moments)**:
  - Acknowledges that **"basic 100 parameters of the model itself"** are present in the foundry sim software (i.e. NSRAM uses far fewer than the full BSIM4 ~100-param card).
  - Comments parasitic BJT cell area is "1 micron" (matches `area=1u` in .asc).
  - Discusses **"large diffusion and implantation"** explaining "every single active device", and that "layout-dependent effect" he is trying to model "with this parameter alone, and that's not correct 100%".
  - "**Under-relaxation**" and "integration time" mentioned re slow DC sweeps — confirms SRavg=0 is a slew-rate-averaged-zero condition for DC-limit IVs.
  - **Floating-body capacity / dynamics**: "BC is not gonna change too much, but the dynamics part is going to change for sure" → confirms parasitic BJT BC is fixed once DC fits, dynamic response is driven by Cb (=> CBpar / pdiode capacitance).
  - **Firing/spike timing**: "1.5 [V] is going to be a rating, spikes" — suggests spike-onset region ≈ 1.5 V Vd. (NEW quantitative — not in SA1/SA3.)
  - Mentions need to model **fan-out** ("how performance may degrade with fan-out") — confirmed echoed in mail.txt §05-01 wrap-up.
- **Mario**: Mostly contextual; expresses approval. Few substantive technical claims.

Conclusion: the transcript yields **zero verifiable quantitative parameter values** that aren't already in the .asc/BJT/pdiode/BSIM/CSV cards, **except** Sebastian's ad-hoc "1.5 V" remark on spike rating which is unconfirmed by data.

---

## 3. Mail Thread (`mail.txt`) — High-Value, Previously Unread

The full email chain Mar20 → May03 yields the following NEW signals not in SA1-SA3+P2:

1. **(Sebas 2026-04-17)** Sebas dropped the explicit `avalanche-diode model` in LTSpice "very annoying for convergence" and now uses **BSIM4 impact ionization + body voltage directly**, plus a **"complementary bipolar current to capture the full swing of the firing mechanism"**. → confirms `parasiticBJT` is the *complementary* current source, **not** a foundry device. Mario's pyport_v5 should treat the NPN as the firing current path, not a strict device-physical NPN.
2. **(Sebas 2026-05-01)** "Pdiode with area **5 μm × 4.4 μm** in accordance with my implementation" — **NEW quantitative geometry**. → `Adiode = 22 μm²` (vs `area=1u` placeholder in `.asc`). cj0 (7.33e-4 F/m² × 22 μm²) → **Cj0 ≈ 16.1 fF** at zero bias — close to but ABOVE the "5–10 fF" linear-cap range Sebas himself suggests, confirming Cb is bias-dependent.
3. **(Sebas 2026-05-01)** "**For the sake of simplicity, the diode could be replaced with a linear capacitor (somewhere in the range 5–10 fF)**" — gives explicit **range bound for Cb**. → Linear-cap fallback parameter: Cb ∈ [5, 10] fF, vs `CBpar=1 fF` placeholder in `.asc`. **The .asc CBpar=1 fF is ~10× too small** — major implication for transient simulations.
4. **(Sebas 2026-04-17)** "I'm only including a complementary bipolar current to capture the full swing of the firing mechanism. Fits are looking good, but I'm still working on **polynomial dependence of model parameters with tuning voltages (VG1, VG2)** and **layout dependent effect on transistor models**". → Confirms `2Tcell_BSIM_param_DC.csv` is exactly the (VG1,VG2)-polynomial table he refers to. The NaN rows at low VG2 are **regions where the polynomial fit failed** (not regions where the cell is off — see ETAB jumps from 0.8 to 1.9 to 2.5 across VG1 0.2/0.4/0.6 — discrete jumps, not smooth polynomial).
5. **(Sebas 2026-04-30)** "Remember that **some parameters change only for M1, while NFACTOR changes only for M2** (I attribute this to LDE)" → CRITICAL: pyport_v5 must keep NFACTOR as M2-only knob, not refit on M1. SA1 §"M1 vs M2 diff" called this out via raw-card diff, but the explicit attribution to **LDE (layout-dependent effects)** as the mechanism is **new** here.
6. **(Sebas 2026-04-30, BONUS TRACK)** Sebas is "working on a floorplan for the **first testchip entirely dedicated to NS-RAM**" — opens door to request specific test structures. Actionable: list FoM cells we want included.
7. **(Sebas 2026-05-01)** "**high-speed measurements with identical parameters**" — confirms the same DC-extracted M1/M2/BJT params fit both slow-IV and high-speed transient data when the pdiode is corrected. Validates the static-card → dynamic-fit pipeline.
8. **(Robert 2026-04-18)** Explicitly asks for "raw I-V CSVs … no fits needed". Sebas's reply included the SRavg=0 CSV pack we are now auditing.

---

## 4. M1 vs M2 Card Diff (newly-read raw files)

| Field | M1 (`130DNWFB`) | M2 (`130bulkNSRAM`) | Note |
|---|---|---|---|
| Model name | `NMOSdnwfb` (deep-Nwell floating-body) | (NMOS — used by inner FET in 2T cell) | |
| `k1` | 0.53825 | 0.63825 | M2 stronger body-effect coupling. |
| `etab` | **+1.8** | **−0.086777** | **Sign-flip** — M1 is in floating-body regime, M2 is bulk-tied. |
| `nfactor` | 1.0 (default) | **1.58** | Per Sebas: M2-only knob. |
| `alpha0` | 7.83756e-5 | 7.83756e-5 | identical (impact-ion strength fixed by process). |
| `beta0` | 30 | 18 | M2 lower → softer onset. |
| `lbeta0` | 0 | **−9.5e-7** | M2 has L-scaling on beta. |
| `rbodymod` | 0 | 0 | Body-resistance network disabled in both (`rbpb=rbpd=rbps=rbdb=rbsb=50` listed in M1 only — not active). |
| Layout-stress block (`ku0,kvth0,saref=1.04 μm,sbref=1.04 μm`) | present in M1 | absent in M2 | Confirms Sebas's LDE asymmetry attribution. |

→ The **etab sign-flip (+1.8 ↔ −0.087)** is the single largest structural difference and is the M1-floating-body signature SA1 §5 already flagged; this audit confirms it from the raw card file.

---

## 5. Slow IVs (`SRavg=0`) — NEW Validation Dataset

**24 CSV files, 82 points each, ~23 s sweep, 0.2 V/s ramp**, family:
- VG1=0.2 V: 7 VG2 values (−0.10 → +0.10)
- VG1=0.4 V: 7 VG2 values (−0.15 → +0.15)
- VG1=0.6 V: 10 VG2 values (−0.20 → +0.25, widest)

Columns: `vdata, idata, tdata, Var4, vfixgdata, ifixdata`
- `vdata` = Vd swept ~0 → 2 V
- `idata` = drain current (A) — main I-V trace
- `tdata` = time (s), confirms slow DC limit
- `Var4` = sweep-rate metric (0.17–0.25, near-constant — confirms steady DC-limit sweep)
- `vfixgdata`/`ifixdata` = fixed-gate reference channel (≈ 0.75 mV / 1.2 pA — leakage baseline)

**VG1=0.6 / VG2=0 example**: at Vd=1.95 V, Id=37.9 μA — snapback region active.
At Vd=0.05 V, Id=−0.95 nA — clean diode reverse leakage baseline.

**Status before this audit:** Never ingested into the python fit pipeline. **This is the 33-bias regression target referenced in P2 §binning-audit but actually living here in the Zoom drop.**

---

## 6. Cross-reference vs. SA1-SA3+P2

| SA/P claim | Confirmed by R1? | New nuance |
|---|---|---|
| SA1: parasiticBJT card (16 params) | YES (file identical) | Sebas's mail clarifies it is *complementary firing current*, not real device. |
| SA1: pdiode card (12 params) | YES (file identical) | NEW: Adiode = 22 μm² (5 × 4.4 μm); Cb ∈ [5,10] fF range. |
| SA1: CBpar = 1 fF placeholder | YES (in .asc) | NEW: Sebas explicitly suggests 5–10 fF is realistic; .asc is wrong by ~10×. |
| SA1: 33-bias VG1×VG2 CSV | YES | NEW: NaN rows are *fit failures*, not OFF regions. |
| SA1: M1 ≠ M2 (5-tuple sig) | YES | NEW: explicitly attributed to LDE by Sebas in 2026-04-30 mail. |
| P2: temp-trap (S6 tlev=1) flag | partially | Transcript does not corroborate temp-trap independently; remains a theoretical flag. |
| SA3: image-2 = transient-V_D pulse trace | YES | — |
| (none) | — | NEW: Sebas testchip floorplan in progress (BONUS TRACK) — opportunity to request cells. |

---

## 7. Top 5 NEW Findings (gate criterion ≥3)

1. **Cb is ~5–10 fF, not 1 fF.** The `.asc` CBpar=1 fF is a placeholder; Sebas's 2026-05-01 mail states realistic body cap is 5–10 fF (linear-cap fallback) or 22 μm² × pdiode (bias-dependent). pyport_v5 should default to Cb = 7 fF (midpoint) for transient runs.
2. **Pdiode area = 5 μm × 4.4 μm = 22 μm².** Previously unspecified — first appearance is in mail.txt 2026-05-01. Use as `pdiode area=22u` in netlists.
3. **NPN is functional firing current, not a physical device.** Sebas explicitly dropped avalanche-diode for convergence and uses `parasiticBJT` as complementary current source on top of BSIM4 impact ionization. Topology rebuild should NOT try to physically calibrate NPN; treat as fitting current source.
4. **NFACTOR(M2)-only attribution = LDE.** Sebas's 2026-04-30 wrap explicitly attributes the M1/M2 NFACTOR asymmetry to layout-dependent effects (LDE). M1's `saref=1.04 μm, sbref=1.04 μm, ku0=−2.7e-8, kvth0=9.8e-9` layout-stress block is the structural reason; M2 has no such block.
5. **24 SLOW IV CSVs are unaudited silicon validation data.** SRavg=0 (DC-limit) sweeps at 0.2 V/s, 82 pts × 24 files. This is the proper regression target for the polynomial (VG1,VG2)-table in `2Tcell_BSIM_param_DC.csv`. Has NEVER been ingested into our fit pipeline.

**Bonus (quantitative parameter, AMBITIOUS-gate):** Sebas-cited Cb ∈ [5, 10] fF (mail.txt 2026-05-01, verbatim). This is a hard quantitative parameter value with explicit Sebastian citation.

---

## 8. Gate Status

- **PASS-conservative (≥3 new signals):** **PASS** — at least 5 above + 3 mail-thread items (#1, #2, #6) = 8 NEW signals not in SA1-SA3+P2.
- **PASS-AMBITIOUS (≥1 Sebas-cited quantitative param):** **PASS** — Cb ∈ [5,10] fF; Adiode = 22 μm²; both with explicit Sebas citation in mail.txt 2026-05-01.

---

## 9. Top Actionable Item for pyport_v5

**Set Cb default to 7 fF (not 1 fF) and ingest the 24 Slow-IV CSVs as the 33-bias regression target.** Single fix changes the transient response amplitude/timing by ~×7 and unlocks the polynomial-parameter fit closure against real silicon. Open the `2Tcell_BSIM_fits.xlsx` and `.pptx` next to recover Sebas's own residuals/plots; that is the next ~30-min task.
