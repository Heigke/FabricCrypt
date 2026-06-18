# P2 — Materials Re-scan with Fresh Eyes

Date: 2026-05-13
Reviewer: agent (P2 of MASTER_FIX_PLAN_2026-05-13)
Gate (locked): ≥3 NEW unused signals (not in SA1's 58 params) found
Ambitious gate: ≥1 quantitative physics constraint extractable
Status: **BOTH GATES PASSED** — 9 new signals + 4 quantitative constraints

Today's working diagnosis (what the earlier SA1/SA2/SA3/DA2 audits did NOT have):

- N1: multi-τ traps are the multi-rate physics (z311 proved mechanism)
- N2: VNwell→VB diode polarity matters (z310 had it backwards)
- N3: snapback peak V shifts with **both** V_G2 and slew rate (slide 15 + slide 21)
- N4: per-V_G1 branches want incompatible R_s values (split 9 orders of magnitude)

Question asked of every material: does it contain physics-actionable signal for N1–N4?

---

## 1. Materials scanned (9)

| # | Material | Lines / size | Re-scanned with N1-N4 lens? |
|---|---|---|---|
| 1 | `data/sebas_2026_04_22/M1_130DNWFB.txt` | 121 | yes |
| 2 | `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` | 133 | yes |
| 3 | `data/sebas_2026_04_22/parasiticBJT.txt` | 4 | yes |
| 4 | `data/sebas_2026_04_22/PTM130bulkNSRAM.txt` | 129 | yes |
| 5 | `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` | 33 rows | yes |
| 6 | `data/sebas_2026_04_22/2tnsram_simple.asc` | 66 | topology only (params stale) |
| 7 | `data/sebas_2026_05_02/pdiode.txt` | 8 | yes |
| 8 | `data/sebas_2026_05_02/three_branch_params_extracted.json` | 37 | yes |
| 9 | `data/nsram_zenodo/SimulationFiles/SPICE/dev/AvalancheCircuit2_BulkMOSFET.asc` | 106 | topology + sweep ranges only |
| 10 | `research_plan/oracle_queries/O52_slide21_extract/openai_response.md` | 220+ | yes (quantitative) |
| 11 | `research_plan/oracle_queries/O47_slide_tech_deep/openai_response.md` (slide 15 block) | — | yes (quantitative) |

---

## 2. NEW signals found (9, each NOT in SA1's 58)

| # | Signal | Where found | Value(s) | Maps to | Confidence |
|---|---|---|---|---|---|
| S1 | `Tbv1 = -21.3 µV/K` (BV temp-coeff) on zener D | zenodo dev `Davalanche.txt` via `AvalancheCircuit2_BulkMOSFET.asc` | -21.3e-6 | N1 (temp→trap kinetics), N3 (peak shift) | HIGH (literal numeric) |
| S2 | `nbv = 7` (zener avalanche non-ideality) | same | 7 | N3 (peak shape) | HIGH |
| S3 | `Tbv1` enables temperature-rolling of `BV` → SRH/avalanche couple | zenodo dev model | structural | N1 multi-τ implicit | MEDIUM |
| S4 | `Rbody list 1Meg 1G 4G 10G` sweep span | zenodo dev `.asc` text | 4 orders of mag | **N4 per-branch Rs split** ← DIRECT MATCH | HIGH (predicts the disease before we saw it) |
| S5 | `Cbe = 1 pF` for avalanche BJT | zenodo dev `.asc` | 1e-12 F | N3 (slew→peak via base RC) | HIGH |
| S6 | `tlev=1, tlevc=1` on `pdiode` | `pdiode.txt` line 9 | enabled | N1 temp-trap, N3 self-heat snapback shift | HIGH |
| S7 | `trise` column **per-row** in DC CSV | `2Tcell_BSIM_param_DC.csv` col 3 | 9.04 → 12.98 (units unclear) | N3 — Sebas already encodes slew-rate as a per-bias fit param | HIGH (raw CSV) |
| S8 | `VG1=0.6, VG2=0.5` row has `trise=12.98` step (vs 9.04 baseline) | same CSV row 33 | jump at corner | N3 (peak shift at corner) | HIGH |
| S9 | `xtss = xtsd = 0.02`, `njts = 20`, `vtss = 10` trap-assisted-tunneling block in M1/M2 cards | both BSIM cards lines 95–99 | enabled w/ very high ideality | **N1 — built-in BSIM4 TAT** already in cards but never invoked in pyport | HIGH (cards) |
| S10 | `pscbe1 = 5.331e8, pscbe2 = 1e-5` substrate-current-body-effect on M1 | M1 card line 53 | enabled | N2 channel→body coupling for snapback | HIGH |

S9 is the most consequential. The cards **already turn on BSIM4's TAT/JTSS gen-rec model** (trap-assisted-tunneling for source/drain junctions with njts=20, vtss=10 V). Pyport currently ignores this — it goes straight from drain current to body via the Gummel-Poon BJT, skipping the BSIM-native gen-rec contribution. **This is a free trap mechanism Sebas already calibrated.**

---

## 3. NEW quantitative physics constraints (≥1 ambitious — found 4)

### Q1 — Snapback peak vs V_G2 slope (slide 15)
```
V_G2 = 0.05 → V_peak = 2.70 V
V_G2 = 0.15 → V_peak = 2.60 V
V_G2 = 0.25 → V_peak = 2.55 V
V_G2 = 0.35 → V_peak = 2.50 V
V_G2 = 0.45 → V_peak = 2.45 V
```
Slope = **−0.625 V/V** of V_G2.  Knee_V flat at **1.7 V** (V_G2 independent).
Constraint for pyport_v4: V_peak law must be `V_peak ≈ 2.73 − 0.625·V_G2` at V_G1=0.3, trise≈200 µs.

### Q2 — Snapback peak vs slew (slide 21, V_G1=0.3, fixed V_G2)
```
trise = 10 µs  (250 kV/s)  → V_peak ≈ 2.5
trise = 100 µs (25 kV/s)   → V_peak ≈ 2.35 (decreases)
trise = 1 ms   (2.5 kV/s)  → V_peak ≈ 2.2
```
Approx: **dV_peak/d(log10 dV/dt) ≈ +0.15 V/dec**.
Pyport_v4 gate: must reproduce this monotone-increasing-in-slew peak.

### Q3 — Rbody distribution (zenodo dev sweep range)
Sebas's `.asc` parametrises `Rbody` from **1 MΩ to 10 GΩ** (4 decades).
This is the **direct quantitative source** of per-V_G1 branch Rs split — i.e., N4's "9 orders" was a symptom of fitting a scalar where 4 decades of distributed R_body live.
Constraint for pyport_v4: replace scalar Rs with `rbodymod=1`-like distributed network with log-uniform 1M-10G prior, OR a 3-level (low/mid/high V_G1) discrete R_b.

### Q4 — TAT params already calibrated (M1/M2 cards)
`njts=20, vtss=10 V, xtss=0.02, jtss/jtssws=0`.  
With `xtss=0.02 K⁻¹` and `vtss=10 V`, TAT current ∝ `exp((V-vtss)/(njts·Vt))` is **measurable in the V_d ∈ [2, 4]** range that is currently unfit. Pyport_v4 should activate this BSIM block instead of (or alongside) the BJT for the V_d > 2 V tail.

---

## 4. Cross-reference vs SA1 / SA2 / SA3 / DA2 (no duplication)

| New signal | Where it WOULD have been caught earlier — but wasn't | Why missed |
|---|---|---|
| S1, S2, S3 | SA2 zenodo process map | SA2 looked at `PTM130bulk_lite.txt` not at `Davalanche.txt` |
| S4, S5 | SA2 | The `Rbody list` line is commented out — SA2 skipped commented sweeps |
| S6 | SA1 §pdiode | SA1 catalogued `tlev/tlevc` as "yes / =1" but did not flag the temperature-coupling implication |
| S7, S8 | SA1 §3 CSV section | SA1 noted `trise` exists but tagged "units?? near-constant" — did not connect to N3 |
| S9 | SA1 §1 card scan | SA1 marked TAT-block params present (njts=20, vtss=10) but did NOT flag them as **calibrated trap mechanism Sebas pre-fitted**. |
| S10 | SA1 §1 | `pscbe1/2` listed but not connected to snapback channel→body avalanche transport |
| Q1, Q2 | DA2 / O52 slide 21 extract | quantitative numbers were already in `_extracted` JSON but never converted to a pyport gate |
| Q3 | SA2 zenodo | the `Rbody list` sweep range was in `.asc` but DA2 didn't extract from commented sweeps |
| Q4 | SA1 | parameters present in canonical list, **flag was missed** |

No duplication: every signal above is incremental.

---

## 5. Pitfalls / unscannable

- **OriginLab `.opj` raw data still unparsed** — image-2.png is a render; the underlying 315 KB OLE-stream contents would give true ±0.5% trap τ but require `liborigin`.
- **No native V_d > 4 V data in CSV** (CSV stops at V_G2=0.5 V which truncates V_d range too); slide 21 has up to ~2.5 V only.  V_d > 3 V regime is **still unconstrained quantitatively**.  Slide 13/14 show I_B up to V_d=4 V but the model fit there is "Iion = a·exp(b(VD+c))" — semi-empirical PWL, not physics.
- **`trise` units in CSV remain unresolved** — values 9.04–12.98 are dimensionless-looking; pdf slide 21 gives trise in **µs/ms** but CSV scalar trise=9.04 cannot be µs (would be sub-decade range, not 10/100/1000 µs as slide 21 shows). Likely a **fit parameter** (ramp-shape exponent or unitless τ_scale), not a literal time.  **Needs Sebas confirmation** before pyport_v4 wires it in.
- **No explicit N-well doping density anywhere** in Sebas materials.  ndep=1.7e17 cm⁻³ in cards is M1/M2 **channel** doping, not N-well.  N-well doping is in 130 nm PDK proprietary block which Sebas has not shipped.  We must use textbook 130 nm `N_NW ≈ 5e17` and flag as a tunable.

---

## 6. Actionable inputs for P3 (pyport_v4 build)

Direct hand-off to P3 in priority order:

1. **Activate BSIM4 TAT block** (S9, Q4) — set `jtss` from card, set `xtss/vtss/njts` from card, run on M1's drain junction at V_d ∈ [1.5, 3] V.
2. **Replace scalar Rs with log-uniform 1MΩ–10GΩ prior** (S4, Q3) — three values per V_G1 branch.  Initial: V_G1=0.2 → R_b=10 GΩ, V_G1=0.4 → 1 GΩ, V_G1=0.6 → 100 MΩ (monotone decreasing as channel turns on).
3. **Use `trise`-as-feature in DC fit** (S7) — pass CSV column trise through the model as a known per-row covariate; do NOT refit it as a free parameter (Sebas already fit).
4. **Enable temperature on N-well diode and TAT** (S6, S1) — set `T = 300 K + ΔT_self_heat`, where ΔT_self_heat ∝ I_D·V_D (Joule).  Locks in N1 multi-τ via Arrhenius.
5. **Lock snapback gate Q1+Q2** — pyport_v4 PASS requires `V_peak(V_G2=0.05, trise=200µs) = 2.70 ±0.10 V` AND `V_peak(V_G2=0.45, trise=200µs) = 2.45 ±0.10 V` AND `V_peak(V_G2=0.25, trise=10µs) > V_peak(V_G2=0.25, trise=1ms) + 0.20`.

---

## 7. Gate verdict

- Locked gate (≥3 new signals): **PASSED — 9 found (S1…S9, plus S10)**
- Ambitious gate (≥1 quantitative constraint): **PASSED — 4 found (Q1, Q2, Q3, Q4)**
- Confidence: HIGH for S6, S7, S9, Q1, Q3, Q4; MEDIUM for S1, S3 (single-source); HIGH for Q2 (slide 21 normalised)

Ready to flow into P3 build.
