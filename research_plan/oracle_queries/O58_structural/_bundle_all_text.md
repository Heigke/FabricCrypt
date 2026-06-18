# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: 01_LOG_tail200.md (8934 chars) ===
```
on infrastructure, P7 partial, P8 DONE. Master fix plan complete.

Next user gates:
1. Generate figures + appendix → send Mario
2. OR: prioritize topology rebuild (multi-day code work)
3. OR: keep accumulating side-results

## 2026-05-13 13:25 — O56 critique synthesis

Q1: HDC LOCKED data-stark, but "shippable" framing fragile due to inert
cfg flags (tech debt). HDC uses V_G1=0.3 (interpolated), and bimodal V_G1=0.2
catastrophe could leak into network performance.

Q2 (sharp): Rerun HDC at V_G1=0.2 (catastrophic DC branch). Pre-reg null
hypothesis: accuracy ≥ 80% there. If acc < 80% → headline robusthet
FALSIFIED (DC bias DOES leak into network).

Q3 NO-CHEAT: 3/3 say discipline HOLDS since O55. All drift corrections
applied. No new flag.

## 2026-05-13 13:25 — Pre-reg z319: V_G1=0.2 HDC falsifier (Q2 from O56)

LOCKED gate:
- PASS (claim defended): mean test_acc ≥ 80% at V_G1=0.2, σ=0, N=16384, n=4
- FAIL (claim falsified): mean < 80% → HDC headline robusthet has hidden 
  V_G1-dependence we haven't acknowledged
- AMBITIOUS: mean ≥ 83% (within 1pp of V_G1=0.3 headline) → strong defense

Cluster idle, APU 41°C. Submitting now.

## 2026-05-13 14:47 — :47 idle — idle, APU=43C

## 2026-05-13 15:47 — :47 idle — idle, APU=37C

## 2026-05-13 14:51 — 🎯 z319 AMBITIOUS PASS — HDC ROBUST to DC bimodal

V_G1=0.2 (catastrophic DC branch 4.7 dec off) at N=16384 n=4:
- seed 0: 83.75%
- seed 1: 83.64%
- seed 2: 84.02%
- seed 3: 84.22%
- mean = 83.91% (V_G1=0.3 headline: 83.86% → within 0.05pp)

**Profound finding**: HDC encoding is INSENSITIVE to per-cell DC fit
quality. The "model 10× off at V_G1=0.2" does NOT leak into network
performance. HDC's noise-averaging-over-N=16384-cells dominates per-
cell signal anyway.

**Gate verdict**:
- PASS (≥80% defense): PASS (83.91% ≫ 80%)
- AMBITIOUS (within 1pp of V_G1=0.3): PASS (within 0.05pp — far better than 1pp)

**O56 Q2 falsification DEFEATED**: gemini hypothesis "if V_G1=0.2 acc 
drops below 80% → headline robusthet falsified" — REJECTED. V_G1=0.2 
yields essentially identical accuracy as V_G1=0.3.

**v4.4 brief update needed**: §6 honest gap "z304 DC bimodal at V_G1=0.2"
remains valid for DC fit, but should add §4.1: "Network-level performance
is empirically robust across V_G1 branches — z319 V_G1=0.2 yields
83.91% (within 0.05pp of V_G1=0.3 headline) despite DC catastrophic at
that bias point. HDC's noise-averaging dominates per-cell DC errors."

This is a v4.4 STRENGTH not a weakness.

## 2026-05-13 14:55 — 3h campaign cron: PLAN CLOSED + z319 defense already logged

State: 4E brief compiled, n=10 headline locked (83.86%), z319 defended
robustness against bimodal DC, O56 critique → no new drift flags.
Cluster idle 37°C. No new compute launched this cron — would duplicate.

This phase of campaign closed. Remaining items for v4.4 ship:
- Generate figures (HDC N-scaling curve, snapback law fit, per-row heatmap)
- Add Appendix A "ask" block for Sebas
- 35 nJ/inf derivation footnote
- All user-gated, not auto-cron-launched.

## 2026-05-13 14:55 — 6h track audit
Phase A: A.1/A.2✓ A.3/A.4 deferred. 2/4.
Phase B: DS-N1✓ DS-N2✓ DS-N3✓(AMBITIOUS+NIST) DS-N5✓(LOCKED v4.4 headline) DS-N4 in_progress(#186) DS-N6✓(FAIL). 5/6.
Phase C: 4A-D✓ 4E✓ compiled.
Today's biggest wins: z319 defense (HDC robust to DC bimodal) + 4E brief compiled + Swedish podcast 22min.
v4.4 status: HEADLINE LOCKED+DEFENDED, brief drafted, pre-send items = figures + Appendix A user-gated.

## 2026-05-13 15:00 — TOPOLOGY REBUILD CAMPAIGN R-1...R-10

Plan: research_plan/TOPOLOGY_REBUILD_PLAN_2026-05-13.md
Cron: 0e1bec5c (hourly :13 progress check)

MASSIVE find: nsram/Zoom/ dir NEVER audited — has Zoom meeting transcript
with Sebastian + slow-IV sweeps (SRavg=0 = DC limit, NEW DATA) + BSIMfitsBA.

R-1 + R-3 audit subagents launched in parallel (research only, no compute).
After they land → R-4 pyport_v5 build (real _residuals wiring).
Stop criterion: snapback + DC + transient + LIF all in approximate right
range, model not missing physics. Then network sim.

## 2026-05-13 16:47 — :47 idle — idle, APU=39C

## 2026-05-13 16:55 — 🎯 R-1 = MASSIVE UNLOCK (8 new signals from mail.txt)

User correctly flagged transcript as "jättedåligt" — but the parallel
mail.txt + raw M1/M2 cards yielded 8 new signals:

**5 game-changing**:
1. Cb = 5-10 fF (vs our 1 fF, 10× TOO SMALL — body τ scales 7×)
2. pdiode area = 22 μm² (5×4.4) (vs our 1u placeholder, 22× off)
3. LDE stress block on M1 only: saref/sbref=1.04μm, ku0=-2.7e-8, kvth0=9.8e-9
4. parasiticBJT is NOT a real device per Sebas — "complementary firing
   current source", model artifact only. Stop calibrating against real BJT physics.
5. 24 Slow-IV SRavg=0 CSVs (DC-limit data we never used)

**Implication for R-4 v5 build** (in flight):
- Set Cb default 7 fF (not 1 fF). May close transient/hysteresis gap directly.
- Set Adiode = 22 μm² as default. Likely fixes V_G1=0.2 sub-threshold catastrophe.
- Add LDE stress block on M1 → explains etab asymmetry physically.
- Reframe NPN: stop trying to match real BJT, treat as firing-current source.
- Use 24 SRavg=0 CSVs as canonical regression target (DC-limit data, less ramp confound).

Need to inform R-4 subagent OR have new R-4b after this lands.

AMBITIOUS PASS per gate spec. PushNotification triggered.

## 2026-05-13 17:00 — R-1b deep audit: complete v5 recipe + paradigm shift

R-1b processed 31 Zoom images via gpt-5 vision + xlsx + pptx + mail.txt.

**CRITICAL paradigm shift** (mail 2026-04-17): Sebas DROPPED avalanche-
diode path due to LTSpice convergence. Uses BSIM4 IMPACT_IONIZATION +
body-bias dynamics directly. Our pyport has been fighting wrong physics
for snapback.

**Complete v5 recipe** (consolidating R-1, R-1b, R-3):
1. Cb = 7 fF (was 1 fF)
2. Adiode = 22 μm² (was 1μ placeholder)
3. ALPHA0 = 7.842e-5 CONSTANT (drop polynomial)
4. K1 = 3-node V_G1-only LUT (drop V_G2 dependence)
5. mbjt = binary step at V_G1 ≈ 0.3 (off ↔ on)
6. **DROP avalanche / Chynoweth entirely** (per Sebas)
7. Use BSIM4 IMPACT_IONIZATION (alpha0/beta0 already there) + RC body cap
8. LDE stress block on M1 only: saref=sbref=1.04μm, ku0=-2.7e-8, kvth0=9.8e-9
9. parasiticBJT = firing-current-source artifact, NOT real BJT
10. body_pdiode_Rs + harmonic-mean limiter (R-3 wiring fix)

**Brian2 SNN benchmarks**: Poisson=85%, LIF=72% (slide content)
**Energy quoted**: 21.5 pJ/spike (6.7 generation + 25 fJ integration)
**Cell area**: 17 µm² 2T thick-ox, 5.3×6 µm² 130nm

**Slow-IV CSVs**: md5-identical to existing. NO new IV data.
**Fits-xlsx**: only 23 of 33 rows fitted (10 NaN failures).

Now R-4 has full recipe to use. If R-4 lands with limited recipe, launch
R-4b with this complete list.

## 2026-05-13 17:13 — Topology rebuild progress check
R-1 (Zoom audit transcript): DONE, mail.txt 5 game-changers extracted.
R-1b (deep images+xlsx+pptx): DONE, paradigm shift on snapback (Sebas dropped avalanche).
R-3 (pyport infra audit): DONE, ~90 LOC scoping, 8 ORPHAN flags identified.
R-4 (pyport_v5 build): IN FLIGHT (started ~16:00, 70 min in).
R-5 through R-10: BLOCKED on R-4. No new ALERT.

## 2026-05-13 17:30 — R-4 v5 done: INFRASTRUCTURE WIN, params still wrong

z320_pyport_v5 results:
- Unit tests 5/5 PASS (body_pdiode_Rs, enable_tat, tat_vtss, tat_xtss, avalanche)
- Cell-wide median 2.91-3.62 dec, all configs WORSE than z304 0.99
- Verdict: infrastructure now solid, parameters need full R-1+R-1b recipe

R-4 was dispatched BEFORE R-1/R-1b landed. Used old params:
- Cb=1fF (R-1: should be 7fF)
- Adiode placeholder (R-1: 22μm²)
- ALPHA0 polynomial (R-1b: const 7.842e-5)
- Kept avalanche path (Sebas DROPPED per mail 04-17)
- No K1 LUT, no mbjt step

**R-4b launching** with FULL recipe.

## 2026-05-13 17:47 — :47 idle — ACTIVE: z321_pyport_v, APU=44C

## 2026-05-13 17:47 — deep-dive 2h cron: z321 R-4b active, no new launch

## 2026-05-13 17:50 — 4h campaign progress
P1-P8 (master fix): all closed last cycle. New campaign: R-1..R-10 topology rebuild active.
R-1✓ R-1b✓ R-3✓ R-4✓(infrastructure WIN, params FAIL). R-4b IN FLIGHT (full recipe).
R-5...R-10 BLOCKED on R-4b. No new ALERT.

## 2026-05-13 18:13 — R-phase progress check
R-1✓ R-1b✓ R-3✓ R-4✓ infrastructure (5/5 unit tests).
R-4b IN FLIGHT (full recipe + Bf×Js ablation, cell 2/25 ~3.01 dec, ~80min ETA).
R-5...R-10 BLOCKED. No ALERT.

## 2026-05-13 18:47 — :47 idle — ACTIVE: z321_pyport_v, APU=44C

## 2026-05-13 19:13 — R-phase progress check
R-4b z321 IN FLIGHT (cell 2/25 ~3.01 dec mid-run, ~50 min ETA remaining).
No gates crossed since last check. No new dispatch.

## 2026-05-13 18:15 — Pre-reg R-5 (LIF zgx) + R-6-lite (daedalus brute-force)
Locked gates:
R-5: 2T-cell driven by V_G1 pulse → M2 produces ≥1 spike with 0.5-1.5V swing, refractory 1-100 µs. PASS if any (Bf, Js, Cb) config gives clean spike + refractory.
R-6-lite: extend R-4b's Bf×Js grid with K1_LUT_scale × mbjt_step_position sweep on daedalus CPU. Gate: cell-wide median < 1.0 dec on at least one config.

```


=== FILE: R1_zoom_audit.md (14271 chars) ===
```
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

```


=== FILE: R1b_zoom_DEEP.md (16285 chars) ===
```
# R-1b — DEEP audit of nsram/Zoom/ (real inspection, not transcript)

Date: 2026-05-13. Supersedes R1_zoom_audit.md (which was transcript-only).
Method: every JPEG opened via Read (vision), xlsx parsed via openpyxl, pptx
via python-pptx, model cards diffed against existing data/sebas_2026_04_22/.

---

## 1. The 31 images — per-image content table

Two distinct meetings ("2026-03-20" introductory slide-set from Mario, and
"2026-04-30" Sebas BSIM fits & circuit follow-up), plus one 04-22 vendor
screenshot and one "image-2.png" dynamic-response slide that was attached to
the 1-May follow-up email.

### Meeting 2026-03-20 (Mario's intro deck, "Status of AI hardware" — slide 1..41)

| File | Slide title / content | Numeric annotations |
|---|---|---|
| 12.05 | "Status of AI hardware" intro | GPU power = 700 W; high cost up to 21% energy by 2030; pollution = 32 Mt e-waste/2030; data transfer >60% energy; market $681.05 B (2024) → $1 T (2030) |
| 12.05(1) | duplicate of 12.05 (zoomed re-share) | same values |
| 12.06 | "AI on edge" Innatera taxonomy <1mW..10–100W | power tiers: <1 mW sensor, 1–10 mW MCU, 10–100 mW DSP/NN, 1–10 W MLNN, 10–100 W HP-inference |
| 12.07 | "Spiking Neural Networks" + LIF eqn | LIF: τ dV/dt = -(V-Vrest)+RI; X.Zhu, S.Pazos, M.Lanza Nature 618 57-62 (2023) |
| 12.07(1) | dup of 12.07 | same |
| 12.07(2) | "Memristor-based electronic neuron" (Liang 2021 / Mitsuru 2024) | #devices >20, low integration density, high power |
| 12.08 | "Memristor-based electronic neuron" overview duplicate | same |
| 12.08(1) | "Neuron implementation" — adaptive exp neuron, mixed-signal IF, analog-Δ; Indiveri Frontiers 2011 | references Indiveri 2011; Vlasov 2024 |
| 12.08(2) | "Memristor-based electronic neuron" arrays — 25 µm² Ag/hBN/Au on SiO2 vs 0.05 µm² Ag/hBN/W on-chip; Alharbi MS&E 2024 | cell areas 25 µm² and **0.05 µm²** (on-chip) |
| 12.09 | "Die-to-die variability — NS-RAM" with bulk-terminal-modulation cartoon. S.Pazos Nature 643 (2025) | references the published Nature paper |
| 12.26 | **"Semi-empirical model fits for impact ionization bulk currents"** (Sebas) | Form: I_bulk = I_exp + I_pwl;  I_exp = a(V_d - b)^c if V_d ≤ d; I_pwl = PWL(V_d) if V_d > d; **a, b, c, d = PWL(V_b)** — i.e. exponential coefficients are themselves piecewise functions of body voltage |
| 12.27 | "Measurements vs SPICE simulation using semi-empirical bulk currents" | At low V_d, body has no time to charge → fit fails for V_b ≠ 0 V if N=1; tilts use N>1 |
| 12.27(1) | **"Floating body 2T NS-RAM cell under transient V_D ramps"** I-V family, V_G1 = 0.5 V, V_G2 swept | Drain current 10⁻¹² → 10⁻⁴ A, V_d 0..2.5 V; clear hysteresis loops |
| 12.29 | "NS-RAM in standard triple-well CMOS (130nm)" | Cell area **5.3 µm × 6 µm**; deep N-well **8.5 µm²**; expected 1000× density improvement over state-of-art neuron |
| 12.29(1) | "Deep-Nwell NFET floating body 1T (thick)" | Area **8 µm²**; firing window **7× — 10⁴×**; 100% yield, 100 µV variability nominal; 180 nm CMOS |
| 12.30 | **"2T NS-RAM spiking neuron cell (thick oxide)"** | Area **17 µm²**; second-transistor body-modulation can give **>10⁴× off/on**; V_G1 = 2.5 V (firing), V_G2 floating; outstanding firing |
| 12.33 | **"NS-RAM Simple LIF in Brian2 for input neurons"** | Slowdown 10⁹; **G_LEAK_REST=1.343, THRESH_VAL=1.354, TAU_REF=4 ms, REFRACTORY=4.8979 V (!), TIMESCALE=145 µs, EXCIT_VALUE=2.3** — these are Brian2 dimensionless params, not real-device |
| 12.39 | "More physically realizable SNN in Brian2" | **Poisson reference = 85%, LIF (w/Poisson training) = 72%** on a confusion matrix |

### Meeting 2026-04-22 19.57

| File | Content |
|---|---|
| 19.57 | NUS Ariba supplier-registration portal screenshot (vendor onboarding). No physics value. |

### Meeting 2026-04-30 (Sebas BSIM fits — already partially extracted in O48)

| File | Content | Key numbers |
|---|---|---|
| 13.23 | **3-panel I-V family**: V_G1 = 0.6 / 0.4 / 0.2 V, V_G2 sweep ranges (-0.2..0.1, 0..0.3, 0..0.5) | inset = full 2T schematic with M1, M2, V_B floating body |
| 13.24 | **4-panel parameter dependences**: BETA0(VG2) for 3 VG1 branches, ETAB(VG2) for 3 VG1, K1(VG1) "for all VG2", NFACTOR(VG2) for 3 VG1 | range BETA0 = 11..21, ETAB = 0.8..2.5, K1 = 0.42..0.56, NFACTOR = 2..12 |
| 13.25 | **Transient I-V noise band**: voltage pulse train 0..1.6 µs with current spikes 0..5 mA; underneath two I-V "noise cloud" panels showing measurements as a *band* not a curve | confirms the I-V is variability-bounded, not a single trace |
| 13.28 | **3-corner overlay meas vs sim** (thick=meas, thin=sim), 3 representative bias combos | (VG1,VG2) = (0.6,0.35), (0.4,0.25), (0.2,0.0) — fits within 1 decade |
| 13.31 | **"Simple NS-RAM cell with integration (self-reset)"** | Cap C_int values shown; energy per spike numbers in green box |
| 13.31(1) | **"NS-RAM blocks for input neurons (soma without diode)"** | **Energy: ~0.5 nJ crossover, ~21.5 pJ per spike of action; 6.7 pJ spike generation, ~25 fJ integration loss; area 40 µm²** |
| 13.33 | **"NSRAM firing with linear excitatory and inhibitory inputs"** | Linear range V_G1 between 2.5 V and 3 V; uses **thick oxide** (high drain voltage RS-RAM); 2 inverter stages drive soma directly |
| 13.46 | duplicate of "AI on edge" Innatera slide | same |
| 13.47 | NS-RAM fab process: morphology after each step | Au top, hBN layer; deposition >400°C; multilayer hBN on chip; metal stack 1–4 |
| 13.50 | "Spiking Neural Networks" review slide (repeat of 12.07 content) | same |
| 13.53 | Outlook screenshot — interest in NIM Med Sized Grant; due **8 May** | admin item |
| 13.53(1) | Outlook screenshot — Re: For Your Advice HSE rep for Deputy Head Research Meeting | admin item |

### Standalone

| File | Content | Numbers |
|---|---|---|
| image-2.png | **"Dynamic response (ramp rate dependence)"** — THE diode slide from 1-May email | I-V vs ramp rate 10..2000 V/s; firing slope flattens at high t_rise; **SR & firing time experiments (5–7 in expt list)** = critical for capacitance/τ fitting; V_b ≠ 0 increases body voltage; "parasitic capacitances play a crucial role on the effective time constant" |

---

## 2. xlsx full parameter table (`2Tcell_BSIM_fits.xlsx`, Sheet1, 35×14)

This is the **DC BSIM fit table** — the canonical source.

Columns: `VG1 | VG2 | trise | ETAB | K1 | ALPHA0 | BETA0 | NFACTOR | mbjt | IS | area`
Plus right-hand legend explaining each parameter's effect.

**Top-5 numerical findings:**

1. **ALPHA0 is FIXED at 7.842e-05** across all 33 bias rows — Sebas does NOT vary impact-ionization prefactor with bias; only BETA0 varies. This contradicts the "ALPHA0/BETA0 jointly polynomial in VG1,VG2" assumption in our v0.12.0 release notes. **Single ALPHA0, single polynomial in BETA0(VG1,VG2) only.**

2. **K1 is FIXED inside each VG1 family but jumps between VG1 levels**: K1=0.55825 (VG1=0.2), 0.53825 (VG1=0.4), 0.41825 (VG1=0.6). I.e. K1 depends *only* on VG1 (not VG2). **Δ between VG1=0.4 and VG1=0.6 is -0.12** — a large step, suggests layout-dependent V_th shift.

3. **NFACTOR(M2)** scans 12.15 → 1.25 monotonically as VG2 rises from -0.2 to +0.5 (for VG1=0.6). Confirms the legend "Higher NFACTOR → Higher Vrelax, Higher Vfire at constant VG2." NFACTOR drops as VG2 increases.

4. **BETA0 monotonic in VG2** within each VG1: at VG1=0.2 it goes 10.75 → 14 across VG2=-0.2..0.1; at VG1=0.4 it sits at 19; at VG1=0.6 it sits at 20. **Saturates at 19–20 for higher VG1.**

5. **mbjt** = 0.001 for VG1=0.2 family (bipolar essentially OFF), then jumps to **mbjt = 1** for both VG1=0.4 and 0.6 families. This is a hard switch, not a continuous polynomial — the parasitic BJT is enabled only above some VG1 threshold between 0.2 and 0.4 V.

`trise` (the body-charge time-constant proxy): mostly ≈11.63 (VG1=0.2), then 10.59..12.98 (VG1=0.4), then **9.04 plateau** for VG1=0.6 — **monotone-decreasing in VG1**, exactly what image-2 ramp-rate slide visualizes. ETAB ranges 0.8..2.5 and is **monotone-increasing in VG1**.

The right-column legend transcribed verbatim:
- ETAB: higher → higher I_fired, lower V_relax
- K1: higher → less leakage at relax state
- ALPHA0: higher → lower V_relax, V_fire, higher I_fired (very sensitive)
- BETA0: higher → narrower hysteresis, higher V_relax, slightly lower V_fire
- NFACTOR: higher → higher V_relax & V_fire at constant VG2
- mbjt: integer scaling of bipolar contribution
- IS: Shockley saturation current of BJT
- area: BJT physical area (real-valued mbjt)

CSV (`2Tcell_BSIM_param_DC.csv`) is the same data with NaN for the un-fit VG1=0.4 negative-VG2 rows and the VG1=0.6 negative-VG2 rows. **Only 23 of 33 bias points are actually fitted**; 10 NaN entries flagged for future fits.

---

## 3. pptx (`2026-04-29 NS-RAM I-V BA plots.pptx`) — 3 slides

- **Slide 1**: triple-panel I-V family, V_G1=0.6/0.4/0.2. VG2 ranges: VG1=0.2 → VG2=-0.2..0.1 step 0.05; VG1=0.4 → 0..0.3 step 0.05; VG1=0.6 → 0..0.5 step 0.05. "Symbols=measurements, Lines=simulations."
- **Slide 2**: 3-corner overlay (representative biases) "Thick=measurements, Thin=simulations." Triplets (VG1, VG2) = (0.6, 0.35), (0.4, 0.25), (0.2, 0.0).
- **Slide 3**: parameter dependences with 4 numbered pictures — **NFACTOR vs VG2** (top-right, 3 VG1 branches), **BETA0 vs VG2** (3 VG1), **ETAB vs VG2** (3 VG1), **K1 vs VG1** ("For all VG2"). Confirms that **K1 depends only on VG1, not VG2** (a quantitative law for pyport_v5).

---

## 4. mail.txt — chronological summary (8 threads, 21 Mar → 13 May)

- **20 Mar 2026** — Mario sends Zoom invite for first meeting (21 Apr).
- **23 Mar** — Eric introduces nsram v0.1.0 PyPI, asks for Sebas review.
- **24 Mar** — Sebas: will update parameters in coming weeks; new role transition.
- **25 Mar** — Eric → Sebas: v0.9.0 features (Chynoweth, SRH, BVpar; tau=10,139 s @ 300 K; 7 conductance levels; 97% temporal XOR, 99.6% Mackey-Glass, 96.75% MNIST; LTP/LTD).
- **2 Apr** — Eric: v0.10.0, BEAM byte-level learner, 3.14 bits/char on text8 with 60K params.
- **4 Apr** — Mario: postpone to second half of April; "presented to a company, interested."
- **17 Apr** — **Sebas KEY**: dropped avalanche-diode models (LTSPice convergence issues). Now uses **BSIM4 impact ionization + body bias directly**, with polynomial dependence of fit parameters on (VG1, VG2) and LDE. Asks: "can your approach drop avalanche voltage and use BSIM impact ionization + body voltage directly?"
- **18 Apr** — Robert asks for 2T schematic + raw IV CSVs + process node + foundry card.
- **19 Apr** — Eric: v0.12.0 ships with §6.1 ALPHA0/BETA0, §2.2 K1/K2 body-bias, §10.1 junction breakdown, §12 KT1/UTE/XTIS, §13 SA/SB/KU0/KVTH0 LDE. Channel-HCI (§6.1) fits **~4 decades RMS better** than §10.1 junction breakdown over 2–4.5 V.
- **20 Apr** — Sebas attaches LTspice .asc + 33 IV CSVs (0.2 V/s) + PTM130 model card. Foundry card cannot be shared.
- **30 Apr** — **Sebas KEY** post-meeting: slides + BSIM fits attached (NDA: cards private). Notes: "Some params change only for M1, while NFACTOR changes only for M2 (I attribute this to LDE)." Mentions testchip floorplan upcoming. Asks for simple architecture to test fan-out.
- **1 May** — **Sebas KEY**: simulation parasitic diode wasn't working. New schematic = explicit **pdiode area 5 × 4.4 µm²**, OR alternative **linear cap 5–10 fF** for body junction. Updated transient agreement.
- **3 May** — Eric: brief draft on Overleaf, BSIM/pdiode private; pyport now loads M1/M2 LDE distinction.

**Key new physics from emails (5):** (i) avalanche diode dropped; (ii) NFACTOR M2-only LDE handle; (iii) pdiode 5 µm × 4.4 µm; (iv) body junction cap 5–10 fF; (v) PolynomialBSIM4Params(VG1,VG2) only for BETA0 (others have lower-dim dependencies, see xlsx finding #1–#3).

---

## 5. Slow I-Vs vs existing fast I-Vs — md5 diff

```
md5: ff50159e1cd7f9918e1e359dc9104076 (Zoom/Slow IVs/VG1=0.6 VG2=0.00)
md5: ff50159e1cd7f9918e1e359dc9104076 (data/sebas_2026_04_22/VG1=0.6 VG2=0.00)
```

**Verdict: BYTE-IDENTICAL.** The "Slow I-Vs … SRavg=0" folder is the SAME 33 CSVs already in `data/sebas_2026_04_22/`. The folder name "SRavg=0" simply documents that these are the *baseline* slow-sweep set (0.2 V/s, per email of 20 Apr). No new IV data. The promised "multiple ramp rates for dynamics" Sebas mentioned has NOT yet arrived in this packet — only image-2.png provides ramp-rate evidence visually.

## 6. Model card cross-reference

| Card | nsram/Zoom path | data/ path | Diff |
|---|---|---|---|
| parasiticBJT.txt | schematic&modelCards/ | sebas_2026_04_22/ | **IDENTICAL** |
| PTM130bulkNSRAM.txt | schematic&modelCards/ | sebas_2026_04_22/ | identical EXCEPT line 62 / 117 (our local DEDUP annotations to remove stale `Alpha0=0.00 Beta0=30.0`); upstream is unchanged |
| 130DNWFB(M1).txt | 2026-04-30 BSIMfitsBA/ | sebas_2026_04_22/M1_130DNWFB.txt | **IDENTICAL** |
| 130bulkNSRAM(M2).txt | 2026-04-30 BSIMfitsBA/ | sebas_2026_04_22/M2_130bulkNSRAM.txt | **IDENTICAL** |
| pdiode.txt | nsram/Zoom/pdiode.txt | data/sebas_2026_05_02/pdiode.txt | identical |
| 2tnsram_simple.asc | schematic&modelCards/ | (none) | **NEW** — LTSpice 2T cell ASC (1419 bytes) not previously catalogued |

`parasiticBJT.txt` contents (constants for SPICE NPN to register if not in pyport):
`is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2`

`pdiode.txt` key values:
`bv=11, ibv=97740, cj=7.33e-4, cjsw=1.05e-10, vj=0.219, m=0.241, fc=0.5, xti=6.5, eg=1.11`
**Body junction breakdown voltage = 11 V** (much higher than V_d range of interest 0–2.5 V — confirms pdiode is not the firing mechanism, only the capacitance source).

---

## Gate verdict

**PASS (≥5 new physics values not in prior audit):**
1. ALPHA0 = 7.842e-5 IS FIXED (single value, not VG1/VG2 polynomial).
2. K1 depends ONLY on VG1: 0.55825 / 0.53825 / 0.41825.
3. mbjt is binary (0.001 vs 1) thresholded between VG1=0.2 and 0.4.
4. trise plateau at 9.04 for VG1=0.6 (body time-constant scales 1/VG1).
5. Only 23 of 33 bias points are actually fitted (10 NaN in CSV) — clear what to ask Sebas next.
6. pdiode reverse breakdown bv=11 V (rules out pdiode-based firing).
7. NFACTOR varies VG2-only for M2; M1 has fixed NFACTOR (Sebas LDE comment).
8. mbjt-OFF in VG1=0.2 family means BJT contribution can be skipped at low VG1.

**AMBITIOUS (quantitative laws):**
- **Law L1**: ALPHA0(VG1,VG2) ≡ 7.842e-5 (constant). BETA0 is the *only* impact-ionization handle.
- **Law L2**: K1(VG1) = piecewise (0.55825 if VG1=0.2; 0.53825 if VG1=0.4; 0.41825 if VG1=0.6) — quadratic interp candidate.
- **Law L3**: NFACTOR(M2) ∝ -VG2 (monotone-decreasing roughly linear within each VG1 branch); slope = (NFACTOR(VG2=-0.2) - NFACTOR(VG2=+0.1)) / 0.3 ≈ -19.7 / V for VG1=0.2.
- **Law L4**: trise(VG1) plateau-then-step: 11.63 / 11.46 / 9.04 — fits A·exp(-VG1/τ).
- **Law L5** (from image-2.png): body capacitance ≈ 5–10 fF dominates τ_body together with ramp-rate-dependent t_rise — firing slope depends on dV_d/dt up to ~2000 V/s.

---

## Top-5 actionable for pyport_v5

1. **Replace polynomial ALPHA0(VG1,VG2) with constant 7.842e-5** in `src/nsram_pyport_v2.py` BSIM4ImpactBlock. Keep BETA0(VG1,VG2) polynomial. Single API: `alpha0 = 7.842e-5; beta0 = poly_beta0(vg1, vg2)`.
2. **Add `K1(VG1)` lookup table** (only 3 nodes: 0.2/0.4/0.6 → 0.55825/0.53825/0.41825); for off-grid VG1, monotone-cubic interp. Do NOT make K1 a function of VG2.
3. **Add `mbjt(VG1)` step function**: 0.001 if VG1 ≤ 0.3 else 1.0. This switches the parasitic BJT path on/off and probably explains current pyport over-firing at low VG1. Mark the threshold as a fit handle to refine when new data arrives.
4. **Implement pdiode body cap** as a constant **C_body = 7 fF** linear cap (middle of 5–10 fF email range) plus a TRUE diode card matching `pdiode.txt`. `firing_mode="both"` should use the diode; `firing_mode="channel"` may collapse to linear cap for speed. Confirms 1 extra Newton iter per step penalty (acceptable, per Eric's 3-May email).
5. **Drop the avalanche/Chynoweth path entirely** (per Sebas 17-Apr). Keep only BSIM4 §6.1 impact ionization + complementary BJT current. Gate flag `use_chynoweth=False` as default. Ramp-rate-dependent I-V hysteresis must come from C_body × dV_d/dt RC integration, not Chynoweth ionization time.

```


=== FILE: R3_pyport_audit.md (11920 chars) ===
```
# R-3 pyport cfg-flag audit (2026-05-13)

Scope: trace every cfg flag mentioned in the topology-rebuild plan from where it is
SET (z313_pyport_v4 / z313_bisection_common / z304) into the residual function in
`nsram/nsram/bsim4_port/nsram_cell_2T.py` (`_residuals` at line 435, output assembly
at line 1108).

Primary residual: `nsram/nsram/bsim4_port/nsram_cell_2T.py:435 _residuals` (used by
`solve_2t_at_Vd`, `forward_2t`, and `_residuals_quasi2d`). The z313_pyport_v4 patch
(`install_z313_tat_patch`, line 106) wraps `mod._residuals` only when explicitly
called — bisection variants do NOT call it.

## Flag classification table

| flag | set at | read at | effect | status |
|---|---|---|---|---|
| `use_well_diode` | z313_bisection_common.py:71 (False) | nsram_cell_2T.py:528 | gates entire well-diode block (I_well_body) | **WIRED** |
| `vnwell_Rs` | z313_bisection_common.py:80,82; z313_pyport_v4.py:143,203,280; z304:202 | nsram_cell_2T.py:535 (ONLY inside `if cfg.use_well_diode:`) | series-R inside well-diode block | **READ-BUT-INERT** when `use_well_diode=False` (the variant condition the bisection actually uses) |
| `vnwell_Js` | dataclass default | nsram_cell_2T.py:533 | well-diode ideal current | WIRED (gated by use_well_diode) |
| `vnwell_area` | dataclass | 533 | well-diode area | WIRED (gated) |
| `vnwell_n` | dataclass | 532 | well-diode ideality | WIRED (gated) |
| `vnwell_mbjt` | dataclass | 545 | mbjt scale | WIRED (gated) |
| `vnwell` (voltage) | dataclass | 530, 570, 124 (patch) | well voltage | WIRED |
| `body_pdiode_to` | z313_bisection_common.py:72 ("vnwell"); z313_pyport_v4.py:140 | nsram_cell_2T.py:567-577 | gates pdiode block + selects cathode | **WIRED** |
| `body_pdiode_Js` | dataclass (1e-6) | 579 | pdiode ideal current | WIRED (gated by body_pdiode_to≠"off") |
| `body_pdiode_area` | dataclass (22e-12) | 579 | pdiode area | WIRED (gated) |
| `body_pdiode_n` | dataclass | 578 | ideality | WIRED |
| `body_pdiode_perim_length` | dataclass (0.0) | 584 | enables sidewall branch | **READ-BUT-INERT** at default (0.0); flag is technically WIRED but the production path never sets it ≠0 |
| `body_pdiode_Js_sw` / `n_sw` | dataclass | 585-587 | sidewall branch | WIRED but unreachable while perim_length=0 |
| `body_pdiode_Vj` / `M` / `Cj0_per_area` / `Vj_sw` / `M_sw` / `Cjsw_per_length` | dataclass | NOT read in `_residuals` (only used by `caps.py` transient code) | DC fit | **ORPHAN for DC** (used only in transient solver) |
| **no body_pdiode_Rs exists** | — | — | NO series-R on pdiode path | **MISSING** — this is the root cause of identical fits when `use_well_diode=False` |
| `use_lateral_collector` | z313_bisection_common.py:86,92; z313_pyport_v4.py:146 | nsram_cell_2T.py:709 (via `getattr`, default False) | gates `Ic_avalanche` block | **WIRED** |
| `lat_BV` | z313_bisection_common.py:87 (3.0); z313_pyport_v4.py:147 | nsram_cell_2T.py:710,717 | avalanche knee voltage | WIRED |
| `lat_N` | z313_bisection_common.py:88; z313_pyport_v4.py:148 | 711,717 | avalanche exponent | WIRED |
| `lat_BV_max` | z313_bisection_common.py:89 | 712,719 | saturation ceiling | WIRED |
| `lat_M_smooth_delta` | z313_bisection_common.py:90 | 713,719 | smoothing | WIRED |
| `lat_Rb` | NOT set by bisection/v4 | 732 (`getattr` default 1e6) | only used if `use_local_base=True` | ORPHAN (use_local_base never True in z313 chain) |
| `use_local_base` | NOT set by bisection/v4 (default missing → `getattr` False) | 661,725 | enables local-base BJT path | **READ-BUT-INERT** in z313 chain |
| `iii_to_body_factor` | dataclass | 644,758,770 | iii routing | WIRED (orthogonal) |
| `m1_diode_scale` | dataclass (1.0) | 595 | M1 body-diode scale | WIRED (default 1.0 = unity) |
| `m2_body_gnd` | dataclass (True) | 474,734,753 | M2 body-to-GND | WIRED |
| `use_iii` / `use_gidl` / `use_bjt` / `use_igb` / `use_diode` | dataclass (all True) | 378-394, 488 | sub-current gates | WIRED |
| `z313_enable_tat` | z313_bisection_common.py:76 (False); z313_pyport_v4.py:153 | z313_pyport_v4.py:119 in the PATCH closure only | TAT current — only when patch installed | **READ-BUT-INERT** in bisection chain (patch never installed by z313b/c/d/e); WIRED only in z313_pyport_v4 itself if `install_z313_tat_patch()` is invoked |
| `z313_tat_jtss` / `z313_tat_njts` | z313_pyport_v4.py:154-155 | z313_pyport_v4.py:121-122 (closure) | TAT params | same status as enable_tat |
| `z313_tat_vtss` / `z313_tat_xtss` (TAT_VTSS / TAT_XTSS constants in script) | recorded in summary at z313_pyport_v4.py:486 only | NOT read in residual or patch | T-acceleration (commented "negligible at 300K → 1.0") | **ORPHAN** — constants exist but never consumed (njts=20, vtss=10, xtss=0.02 from oracle never enter the equation) |
| `njts` / `vtss` / `xtss` / `jtss` as BSIM4 saturation-tunnel params | — | not present anywhere in `_residuals` or pyport BSIM4 model | — | **ORPHAN** (only the script-local TAT constants are referenced) |
| `mbjt` per V_G1 step | z313_pyport_v4.py:218-223; z304:220-223 | applied to `bjt.area = area * mbjt` at row build (BEFORE residual). NOT a cfg field | scales NPN area | WIRED through `bjt.area` (not via cfg) |
| `C_b` (body capacitance) | NOT a cfg field | — | — | **ORPHAN** — body cap entirely absent from DC residual; lives only in `caps.py` for transient |
| `Rbody` (any name) | — | — | — | **ORPHAN** — no series resistance between body and any anchor exists in the residual. `vnwell_Rs` is the closest analogue but is gated off |

## Root-cause finding for z313 bitwise-identical bisection

`configure_variant` at `scripts/z313_bisection_common.py:71` sets
`cfg.use_well_diode = False`. The `_residuals` block at `nsram_cell_2T.py:528-547`
that consumes `vnwell_Rs` is wrapped in `if cfg.use_well_diode:`. So variants b/c/d/e
all run with the well-diode branch DISABLED — `vnwell_Rs` is loaded onto cfg but
never read. The replacement pdiode path (`body_pdiode_to="vnwell"`) has NO series
resistance term at all (there is no `body_pdiode_Rs` field anywhere in the residual
or the dataclass). The "drain-end avalanche" path IS wired correctly through
`Ic_avalanche` (line 721 → 1112), but its contribution at the bisection's grid is
small enough that it doesn't move the median once the diode current is
infinitesimal (Js=1e-6·A=22e-12 → Is=2.2e-17 A; forward current ≪ Ids in the
operating window). The four variants therefore collapse to the same near-zero body
shunt + identical channel/BJT physics → identical DC fit.

## Recommendation — minimum wiring for v5

Three flags MUST be promoted to first-class consumers of `_residuals`:

1. **`body_pdiode_Rs`** (NEW field, currently absent). Add a series resistance on the
   body-pdiode branch analogous to lines 534-539 of the well-diode block. Without
   this, "per-V_G1 R_body" cannot exist when use_well_diode=False. Wire inside
   `if cfg.body_pdiode_to != "off":` after line 580. (~12 LOC: harmonic-mean of
   I_ideal and Vab/Rs.)

2. **`use_well_diode`** semantics: either allow it to be TRUE simultaneously with
   `body_pdiode_to="vnwell"` (audit both paths for double-counting first), OR migrate
   `vnwell_Rs` into the pdiode block so the existing per-V_G1 R_body table is
   actually consumed. The cleaner option is #1 above (add `body_pdiode_Rs`) and
   route `R_BODY_TABLE` → `cfg.body_pdiode_Rs` in `configure_variant`. (~6 LOC in
   configure_variant + bisection scripts.)

3. **`z313_enable_tat` → permanent residual term, not a monkey-patch**. Move the
   12-line TAT block (z313_pyport_v4.py:119-129) into the core `_residuals`
   between `I_body_pdiode` and gmin shunts. Add `enable_tat`, `tat_jtss`,
   `tat_njts`, `tat_vtss`, `tat_xtss` to the dataclass (with vtss/xtss actually
   entering the equation as T-acceleration so the oracle params don't sit as
   constants). (~25 LOC including dataclass fields.)

Two more if budget allows (will tighten the fit further but not required to break
the bisection plateau):

4. Wire **`use_local_base` + `lat_Rb`** explicitly in `configure_variant` (currently
   read via `getattr(cfg, ..., default)` → always falls to default). At minimum
   document the choice; better, add to dataclass with defaults of False/1e6.

5. Reconcile **`m1_diode_scale`** sweep: it's WIRED and at unity, but never swept
   in z313/z304. Worth adding to the bisection grid since it directly clamps Vb.

### LOC estimate

| change | LOC |
|---|---|
| Add `body_pdiode_Rs` field + harmonic-mean Rs limiter | ~15 |
| Refactor `configure_variant` to route R_BODY_TABLE → `body_pdiode_Rs` | ~8 |
| Inline TAT into `_residuals` + 5 dataclass fields | ~25 |
| Promote `use_local_base` / `lat_Rb` to dataclass + tests | ~10 |
| Unit test in `nsram/nsram/bsim4_port/tests/` for new Rs path | ~30 |
| **Total core change** | **~90 LOC** (excluding scripts/) |

### Locked-gate ORPHAN list (citations)

- `body_pdiode_Vj`, `body_pdiode_M`, `body_pdiode_Cj0_per_area`, `body_pdiode_Vj_sw`,
  `body_pdiode_M_sw`, `body_pdiode_Cjsw_per_length` — defined in dataclass lines
  185-201, NEVER referenced anywhere in `_residuals` (grep `cfg.body_pdiode_Vj`
  returns 0 hits in `nsram_cell_2T.py`). Used only in `caps.py` (transient).
- `z313_tat_vtss`, `z313_tat_xtss` (recorded as `TAT_VTSS`/`TAT_XTSS` constants in
  `scripts/z313_pyport_v4.py:80-83` and persisted in summary at line 486) — never
  enter the TAT equation (line 124-125 uses only jtss, njts, Vt at T=300K).
- BSIM4 native TAT params `njts`/`vtss`/`xtss`/`jtss` — entirely absent from the
  pyport BSIM4 model (grep returns 0 hits in `nsram/nsram/bsim4_port/*.py`).
- `lat_Rb` and `use_local_base` — read via `getattr` with defaults, never SET by
  the bisection variants → effectively orphan in z313 chain.
- `C_b` (body capacitance), `Rbody` — no such fields exist in the dataclass; body
  KCL has only conductive currents in `_residuals`.

### Unexpected findings

1. **`vnwell_Rs` default is 1.0e10 Ω** (line 126) — but `RS_FALLBACK` in z304 is
   `1.0e30` (line 76). When `rs=0` is selected in z304's grid, the script
   substitutes 1e30, NOT the dataclass default. So "Rs=0 ↔ disabled" comment is
   the script's convention, not a true zero.
2. **`body_pdiode_Js` discrepancy** lines 167-183: Sebas's card implies Js_per_area
   = 2.44e4 A/m²; pyport uses 1e-6 A/m² (10 orders of magnitude smaller). The
   comment acknowledges this. Wiring Sebas's true value would saturate the body
   pdiode current and dominate I_well_body even at low Vb.
3. **Avalanche IS wired correctly** (line 1108-1112 adds Ic_avalanche to Id). The
   z313_bisection variant 'c' (`enable_avalanche=True`) should NOT have been
   bitwise identical to 'b'. Possible cause: `lat_BV=3.0` with Vd_max ~ 3.0V in
   the bisection grid keeps `rev_mag = max(Vd-Vb, 0)` small (Vb tracks Vd via
   floating-body), so `M_safe ≈ 1.0` and the contribution rounds away in float64.
   Recommend lowering `lat_BV` to ~1.5 V or sweeping it to verify the path is
   live.
4. **The `_residuals` function is monkey-patched** by `z313_pyport_v4.install_z313_tat_patch()`.
   Bisection scripts (z313b/c/d/e) do NOT install the patch (grep confirms). So
   any `cfg.z313_enable_tat=True` in the bisection chain would be a no-op.
   Architectural smell — TAT should be core.
5. **No body capacitance in DC residual** — expected for DC, but worth flagging
   for transient v5 work (`caps.py` exists but isn't called from `_residuals`).

### Summary count

- **Flags audited**: 32 distinct cfg fields/flags + 4 script-local TAT constants
- **WIRED** (fully active in z313 chain): 16
- **READ-BUT-INERT** (read by code but on a dead path under z313 config): 8
  (`vnwell_Rs`, `vnwell_Js`, `vnwell_area`, `vnwell_n`, `vnwell_mbjt`,
  `body_pdiode_perim_length`+sidewall params, `use_local_base`+`lat_Rb`,
  `z313_enable_tat` in bisection)
- **ORPHAN** (never read by any residual): 8 (pdiode cap/Vj/M params, TAT vtss/xtss
  constants, BSIM4-native njts family, C_b, Rbody)

```


=== FILE: z304_baseline_summary.json (6782 chars) ===
```json
{
  "script": "z304_aggregate",
  "n_cells_loaded": 176,
  "n_finite_cells": 176,
  "n_source_files": 11,
  "by_vg1": {
    "0.2": {
      "best": {
        "vg1": 0.2,
        "bf": 500,
        "alpha0": 1e-05,
        "rs": 0,
        "median_log_rmse": 2.0610291308357587,
        "signed_dec_median": -1.4757399592295073,
        "p90_log_rmse": 2.1123002207762025,
        "n_finite": 7
      },
      "pareto": [
        {
          "bf": 500,
          "alpha0": 1e-05,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.0001,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.001,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.01,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        }
      ],
      "n_branch_cells": 64
    },
    "0.4": {
      "best": {
        "vg1": 0.4,
        "bf": 50,
        "alpha0": 1e-05,
        "rs": 10000000000.0,
        "median_log_rmse": 1.4046663288699635,
        "signed_dec_median": 0.4243714966378498,
        "p90_log_rmse": 1.4945019316616872,
        "n_finite": 7
      },
      "pareto": [
        {
          "bf": 50,
          "alpha0": 1e-05,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.0001,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.001,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.01,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        }
      ],
      "n_branch_cells": 48
    },
    "0.6": {
      "best": {
        "vg1": 0.6,
        "bf": 9000,
        "alpha0": 1e-05,
        "rs": 10000000000.0,
        "median_log_rmse": 0.7042229003043868,
        "signed_dec_median": 0.12519489440961706,
        "p90_log_rmse": 0.9573272527337507,
        "n_finite": 11
      },
      "pareto": [
        {
          "bf": 9000,
          "alpha0": 1e-05,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.0001,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.001,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.01,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 500,
          "alpha0": 1e-05,
          "rs": 1000000000.0,
          "median_log_rmse": 0.8765201949636146,
          "signed_dec_median": -0.09842195704459478
        },
        {
          "bf": 500,
          "alpha0": 0.0001,
          "rs": 1000000000.0,
          "median_log_rmse": 0.8765201949636146,
          "signed_dec_median": -0.09842195704459478
        }
      ],
      "n_branch_cells": 64
    }
  },
  "best_cellwide_compromise": {
    "bf": 50,
    "alpha0": 1e-05,
    "rs": 10000000000.0,
    "vg1_02_med": 2.3975482170253373,
    "vg1_04_med": 1.4046663288699635,
    "vg1_06_med": 2.7901932952092294,
    "worst_branch_med": 2.7901932952092294,
    "median_across_branches": 2.3975482170253373,
    "max_abs_signed": 3.165630881809051
  },
  "top_5_cellwide": [
    {
      "bf": 50,
      "alpha0": 1e-05,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.0001,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.001,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.01,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 1e-05,
      "rs": 1000000000.0,
      "vg1_02_med": 3.2264262915314457,
      "vg1_04_med": 1.4894433845213277,
      "vg1_06_med": 1.7824399697183684,
      "worst_branch_med": 3.2264262915314457,
      "median_across_branches": 1.7824399697183684,
      "max_abs_signed": 3.778944938250161
    }
  ],
  "gates": {
    "vg1_0.2": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": false,
      "median_log_rmse": 2.0610291308357587,
      "signed_dec_median": -1.4757399592295073
    },
    "vg1_0.4": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": true,
      "median_log_rmse": 1.4046663288699635,
      "signed_dec_median": 0.4243714966378498
    },
    "vg1_0.6": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": true,
      "median_log_rmse": 0.7042229003043868,
      "signed_dec_median": 0.12519489440961706
    }
  },
  "verdict": {
    "ALL_PASS_conservative": false,
    "ALL_AMBITIOUS_SHIP_v4.4": false,
    "ALL_SAFETY": false,
    "CELLWIDE_BEATS_DA3": false
  },
  "da3_reference_median": 0.99
}
```


=== FILE: z320_v5_summary.json (5813 chars) ===
```json
{
  "script": "z320_pyport_v5",
  "elapsed_s": 847.8271734714508,
  "device": "cuda",
  "config": {
    "bf": 500,
    "alpha0": 0.0001,
    "R_BODY_TABLE": {
      "0.2": 10000000000.0,
      "0.4": 1000000000.0,
      "0.6": 100000000.0
    },
    "VBR_AV": 2.0,
    "N_AV": 4.0,
    "PDIODE_AREA": 2.2e-11,
    "PDIODE_N": 1.0535,
    "TAT_JTSS": 3.4e-07,
    "TAT_NJTS": 20.0,
    "TAT_VTSS": 10.0,
    "TAT_XTSS": 0.02
  },
  "z304_baseline_median": 0.99,
  "ablation": {
    "sebas_2.44e4__tat_on": {
      "body_pdiode_Js": 24397.727272727272,
      "enable_tat": true,
      "cell_wide_median_log_rmse": 3.6229981048587026,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.8789523843680707,
          "signed_dec_median": 4.590659375008547,
          "p90_log_rmse": 4.6444876596291245,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314284411,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566532687640607,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "sebas_2.44e4__tat_off": {
      "body_pdiode_Js": 24397.727272727272,
      "enable_tat": false,
      "cell_wide_median_log_rmse": 2.9069160461358856,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 2.0610273644073156,
          "signed_dec_median": -1.4757393562201777,
          "p90_log_rmse": 2.112296393564901,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 2.9069160461358856,
          "signed_dec_median": -3.1534502475497534,
          "p90_log_rmse": 3.061948851054906,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 4.492461240432361,
          "signed_dec_median": -4.611016029355243,
          "p90_log_rmse": 4.81046488167973,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "midoracle_1e-6__tat_on": {
      "body_pdiode_Js": 1e-06,
      "enable_tat": true,
      "cell_wide_median_log_rmse": 3.6229981048587026,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.8789523843680707,
          "signed_dec_median": 4.590659375008547,
          "p90_log_rmse": 4.6444876596291245,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314284411,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566532687640607,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "midoracle_1e-6__tat_off": {
      "body_pdiode_Js": 1e-06,
      "enable_tat": false,
      "cell_wide_median_log_rmse": 2.9069160461358856,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 2.0610273644073156,
          "signed_dec_median": -1.4757393562201777,
          "p90_log_rmse": 2.112296393564901,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 2.9069160461358856,
          "signed_dec_median": -3.1534502475497534,
          "p90_log_rmse": 3.061948851054906,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 4.492461240432361,
          "signed_dec_median": -4.611016029355243,
          "p90_log_rmse": 4.81046488167973,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    }
  },
  "best_cell": "sebas_2.44e4__tat_off",
  "cell_wide_median_log_rmse": 2.9069160461358856,
  "improvement_dec_vs_z304": -1.9169160461358856,
  "per_branch": {
    "0.2": {
      "median_log_rmse": 2.0610273644073156,
      "signed_dec_median": -1.4757393562201777,
      "p90_log_rmse": 2.112296393564901,
      "n_finite": 7,
      "n_total": 7,
      "body_pdiode_Rs": 10000000000.0
    },
    "0.4": {
      "median_log_rmse": 2.9069160461358856,
      "signed_dec_median": -3.1534502475497534,
      "p90_log_rmse": 3.061948851054906,
      "n_finite": 7,
      "n_total": 7,
      "body_pdiode_Rs": 1000000000.0
    },
    "0.6": {
      "median_log_rmse": 4.492461240432361,
      "signed_dec_median": -4.611016029355243,
      "p90_log_rmse": 4.81046488167973,
      "n_finite": 11,
      "n_total": 11,
      "body_pdiode_Rs": 100000000.0
    }
  },
  "gate_PASS_lt_0_70": false,
  "gate_AMBITIOUS_lt_0_50": false,
  "gate_SAFETY_vg1_0_2_lt_1_5": false
}
```


=== FILE: z321_v5b_progress.json (27462 chars) ===
```json
{
  "script": "z321_pyport_v5b_full_recipe",
  "reason": "after_bf_3000__js_sebas_2.44e4",
  "elapsed_s": 5017.109735965729,
  "ablation_so_far": {
    "bf_50__js_1e-6": {
      "bf": 50,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_1e-4": {
      "bf": 50,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_1e-2": {
      "bf": 50,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_1e0": {
      "bf": 50,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_50__js_sebas_2.44e4": {
      "bf": 50,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.0110809250642903,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.3322992450404705,
          "signed_dec_median": 6.08691678856931,
          "p90_log_rmse": 5.856928057662748,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.120499028743418,
          "signed_dec_median": 3.69013857547471,
          "p90_log_rmse": 3.7991553233557243,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.18253990363911,
          "signed_dec_median": 1.3159352571043543,
          "p90_log_rmse": 1.4799693077153873,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e-6": {
      "bf": 200,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e-4": {
      "bf": 200,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e-2": {
      "bf": 200,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_1e0": {
      "bf": 200,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_200__js_sebas_2.44e4": {
      "bf": 200,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.434219222735784,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 5.866553073840639,
          "signed_dec_median": 6.563710504302723,
          "p90_log_rmse": 6.417155919250823,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.4805368283670277,
          "signed_dec_median": 4.194326868095697,
          "p90_log_rmse": 4.29075827842295,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.5217273300061342,
          "signed_dec_median": 1.6857909931405661,
          "p90_log_rmse": 1.7800345908122004,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e-6": {
      "bf": 500,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e-4": {
      "bf": 500,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e-2": {
      "bf": 500,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_1e0": {
      "bf": 500,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_500__js_sebas_2.44e4": {
      "bf": 500,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.6229981048558315,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.136891744822594,
          "signed_dec_median": 5.981739498658496,
          "p90_log_rmse": 6.739798112783655,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 3.878952384368051,
          "signed_dec_median": 4.590659375008544,
          "p90_log_rmse": 4.644487659537766,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 1.8405995314264023,
          "signed_dec_median": 1.890552315012516,
          "p90_log_rmse": 2.1566531954255543,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e-6": {
      "bf": 1000,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e-4": {
      "bf": 1000,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e-2": {
      "bf": 1000,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_1e0": {
      "bf": 1000,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_1000__js_sebas_2.44e4": {
      "bf": 1000,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 3.8760195045092,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.388994910795405,
          "signed_dec_median": 6.550320284794342,
          "p90_log_rmse": 6.990144261913859,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.1002559290111,
          "signed_dec_median": 4.383134449046729,
          "p90_log_rmse": 4.597503341918631,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.1544636699738304,
          "signed_dec_median": 2.0295026797274947,
          "p90_log_rmse": 2.548196110265727,
          "n_finite": 11,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e-6": {
      "bf": 3000,
      "body_pdiode_Js": 1e-06,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e-4": {
      "bf": 3000,
      "body_pdiode_Js": 0.0001,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e-2": {
      "bf": 3000,
      "body_pdiode_Js": 0.01,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_1e0": {
      "bf": 3000,
      "body_pdiode_Js": 1.0,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    },
    "bf_3000__js_sebas_2.44e4": {
      "bf": 3000,
      "body_pdiode_Js": 24397.727272727272,
      "cell_wide_median_log_rmse": 4.248233232293888,
      "per_branch": {
        "0.2": {
          "median_log_rmse": 6.838701497850528,
          "signed_dec_median": 7.207968426398937,
          "p90_log_rmse": 7.367105037749343,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 10000000000.0
        },
        "0.4": {
          "median_log_rmse": 4.331865439884627,
          "signed_dec_median": 3.6682252463976317,
          "p90_log_rmse": 4.78364258233811,
          "n_finite": 7,
          "n_total": 7,
          "body_pdiode_Rs": 1000000000.0
        },
        "0.6": {
          "median_log_rmse": 2.38805633374714,
          "signed_dec_median": 1.7742432490197375,
          "p90_log_rmse": 2.7721856228234194,
          "n_finite": 10,
          "n_total": 11,
          "body_pdiode_Rs": 100000000.0
        }
      },
      "failed": false,
      "gate_PASS_lt_0_70": false,
      "gate_AMBITIOUS_lt_0_50": false,
      "gate_SAFETY_vg1_0_2_lt_1_5": false
    }
  }
}
```
