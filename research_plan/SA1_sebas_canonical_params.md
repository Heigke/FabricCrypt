# SA1 — Sebas Canonical Parameter Reference

**Authoritative consolidation of every parameter Sebas has delivered for the 130 nm 2T NS-RAM cell.**
Date compiled: 2026-05-12. Sources: `data/sebas_2026_04_22/` (M1, M2, parasiticBJT, 2Tcell_BSIM_param_DC.csv, .asc), `data/sebas_2026_05_02/` (pdiode, three_branch_params_extracted.json, image-2.png).

---

## 1. Source inventory & provenance

| File | Date | Role | Format |
|---|---|---|---|
| `M1_130DNWFB.txt` | 2026-04-30 | M1 BSIM4 deep-Nwell floating-bulk NMOS card (`NMOSdnwfb`) — top transistor of 2T cell | BSIM4 v4.5, Level 14 |
| `M2_130bulkNSRAM.txt` | 2026-04-30 | M2 BSIM4 bulk NMOS card (`NMOS`) — bottom (gating) transistor | BSIM4 v4.5, Level 14 |
| `parasiticBJT.txt` | 2026-04-22 | NPN parasitic from D→Sint→B path (floating-body bipolar) | Gummel-Poon `.model NPN` |
| `pdiode.txt` | 2026-05-02 | Body-tap p-diode (Sint↔Nwell or substrate junction) | Generic level=1 diode |
| `PTM130bulkNSRAM.txt` | 2026-04-30 | PTM 130nm reference card (legacy, single-set) | BSIM4 |
| `2Tcell_BSIM_param_DC.csv` | 2026-04-30 | Sebas's **DC sweep table** — per-(V_G1,V_G2) BSIM overrides for fitting | CSV |
| `three_branch_params_extracted.json` | 2026-05-11 | **Manual PNG-extraction** from Sebas's 30-Apr slide deck (NFACTOR/ETAB/K1/BETA0 vs V_G2 per branch) | JSON, ±5% / ±0.01 V |
| `image-2.png` | 2026-05-02 | Dynamic ramp-rate slide (see §6) | PNG |
| `2tnsram_simple.asc` | 2026-04-22 | LTSpice schematic | Netlist |
| `2vHCa-2 I-Vs@VG2 VG1={0.2,0.4,0.6} vnwell=2/` | 2026-04-22 | Raw measured I-V (silicon ground truth, 33-bias grid) | CSV per (V_G1,V_G2) |

**Process markers (M1 + M2):** BSIM4 v4.5, Level 14, `toxn = 4e-9 m` (≈4.0 nm gate-ox, thick-ox 130 nm flavor), `xj = 1.5e-7`, `ndep = 1.7e17`, `lpe0n = 1.244e-7`, `vth0n = 0.54153`. Tnom = 25 °C. Reference comments cite `*+Vth0 = 0.395 Rdsw = 200` (stock PTM 130 nm) but the active cards use `vth0n = 0.54153` (Sebas-tuned, ~+150 mV).

**M1 vs M2 distinguishing features (only places they differ):**
| Param | M1 (DNWFB) | M2 (bulk) | Comment |
|---|---|---|---|
| Model name | `NMOSdnwfb` | `NMOS` | Floating-body vs bulk |
| `k1` | 0.53825 | 0.63825 | M1 body-effect weakened (floating bulk) |
| `etab` | **+1.8** | −0.086777 | M1 sign-flipped & 20× — the floating-body signature |
| `beta0` | 19 | 18 | Avalanche/II generation strength |

Everything else (mobility u0, vsat, dvt*, voff, nfactor base, k2, k3, ua/ub/uc, etc.) is identical across M1 and M2 — they share the underlying PTM 130 nm process and only diverge in three lines.

---

## 2. Per-branch DC table (CSV — authoritative for SA4)

`2Tcell_BSIM_param_DC.csv` is the single most actionable file. It enumerates **31 bias points** with the BSIM overrides Sebas uses at each:

- **Columns:** `VG1, VG2, trise, ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS, area`
- **V_G1 grid:** {0.2, 0.4, 0.6} V (3 branches)
- **V_G2 grid:** −0.20…+0.50 V, step 0.05 V (branch-dependent extent — see below)
- **NaN regions:** V_G1=0.4 and 0.6 branches have NaN for V_G2 ∈ [−0.20, −0.05] (transistor sub-cutoff, no fit attempted)
- **Globals (constant across all rows):** `ALPHA0 = 7.842e-5`, `IS = 5e-9`, `area = 1e-6`
- **K1 collapses to one value per V_G1 branch** (matches JSON `K1_M1_vs_VG1` claim "for all V_G2"):
  - V_G1=0.2 → K1=0.55825
  - V_G1=0.4 → K1=0.53825
  - V_G1=0.6 → K1=0.41825
- **mbjt (BJT area multiplier):** 0.001 on V_G1=0.2 branch, **1.0** on V_G1=0.4 and 0.6 branches → at low V_G1 the parasitic BJT is suppressed 1000× by Sebas's fit; turns on fully above V_G1≈0.3 V.
- **trise (transient ramp parameter):** ~9–13 (units?? — likely ns or normalized; near-constant per branch except a step at V_G2=0.5 on the 0.6 branch).

### V_G2 coverage by branch (after NaN removal)
| V_G1 | V_G2 min | V_G2 max | # points |
|---|---|---|---|
| 0.2 | −0.20 | +0.10 | 7 |
| 0.4 | 0.00 | +0.30 | 7 |
| 0.6 | 0.00 | +0.50 | 11 |

---

## 3. Reconciliation: CSV vs JSON-from-PNG

The JSON is a re-reading of slide plots that were themselves rendered from the same CSV. Cross-check:

| Param | CSV value at (V_G1, V_G2) | JSON inferred | Match? |
|---|---|---|---|
| NFACTOR @ (0.2, −0.20) | 12.15 | 12.20 (red) | ✓ (Δ0.05) |
| NFACTOR @ (0.2, 0.10) | 6.25 | 6.25 (red) | ✓ exact |
| NFACTOR @ (0.6, 0.00) | 6.00 | 6.00 (black) | ✓ exact |
| NFACTOR @ (0.6, 0.50) | 1.25 | 1.25 (black) | ✓ exact |
| NFACTOR @ (0.4, 0.10) | 5.00 | 5.00 (blue) | ✓ exact |
| ETAB @ (0.2, 0.00) | 1.00 | 1.00 (red) | ✓ |
| ETAB @ (0.6, 0.00) | 2.50 | 2.50 (black) | ✓ |
| ETAB @ (0.4, 0.00) | 1.90 | 1.90 (blue) | ✓ |
| BETA0 @ (0.2, 0.00) | 12.5 | 12.5 (red) | ✓ |
| BETA0 @ (0.4, ≥0) | 19 (flat) | 19 (blue) | ✓ |
| BETA0 @ (0.6, ≥0) | 20 (flat) | 20 (black) | ✓ |
| K1 @ V_G1=0.2 | 0.55825 | 0.558 | ✓ |
| K1 @ V_G1=0.4 | 0.53825 | 0.538 | ✓ |
| K1 @ V_G1=0.6 | 0.41825 | 0.418 | ✓ |

**Result:** **No conflicts.** JSON branch colour mapping confirmed: red→V_G1=0.2, blue→V_G1=0.4, black→V_G1=0.6. The JSON `_branch_assignment_note` warning can be RESOLVED — colors are correct.

### CSV vs M1/M2 card defaults
| Param | M1 card | M2 card | CSV uses | Interpretation |
|---|---|---|---|---|
| `k1` | 0.53825 | 0.63825 | 0.418–0.558 (per branch) | CSV K1 **does NOT match M2's 0.63825** — Sebas's fit overrides M2 K1 per branch. M1 default 0.53825 ≈ V_G1=0.4 branch. |
| `etab` | +1.8 | −0.086777 | 0.8 → 2.5 | CSV ETAB matches M1 sign/magnitude (positive, O(1)). M2's −0.086777 is irrelevant — ETAB is an M1-floating-body parameter. |
| `beta0` | 19 | 18 | 10.75 → 20 | CSV BETA0 spans the M1=19 / M2=18 region. CSV is per-(V_G1,V_G2) override of M1.beta0. |
| `nfactor` | 1.58 | 1.58 | 1.25 → 12.15 | CSV NFACTOR is **8× larger** than card default — the parasitic-BJT-dominated subthreshold regime needs much larger N to fit slope. |
| `alpha0` | 7.83756e-5 | 7.83756e-5 | 7.842e-5 | ✓ matches (Sebas froze ALPHA0). |

**Critical M3 observation (confirmed):** NFACTOR ranges up to **12.2** (V_G1=0.2, V_G2=−0.2). BBO/optimiser bound of 3.0 used in our prior M3 fits cuts off the lower-V_G1 cold-bias regime. Refit must allow NFACTOR ∈ [1, 15].

**Structural finding (confirmed):** BETA0 is flat-then-step-then-flat across V_G1 branches (10.75→14 then 19 then 20). No smooth polynomial in (V_G1, V_G2) can represent this — **branch decomposition is mandatory** for BETA0 (and arguably for K1 and mbjt as well).

---

## 4. Canonical parameter table

| param | value(s) | source | confidence | scope | notes |
|---|---|---|---|---|---|
| **Process / global (frozen)** | | | | | |
| toxn | 4e-9 m | M1, M2 (`.param`) | high (card) | global | gate oxide; thick-ox 130 nm flavor |
| xj | 1.5e-7 m | M1, M2 | high | global | junction depth |
| ndep | 1.7e17 cm⁻³ | M1, M2 | high | global | body doping |
| vth0n | 0.54153 V | M1, M2 (`.param`) | high | global | Sebas-tuned (+150 mV vs PTM stock 0.395) |
| ALPHA0 | 7.842e-5 (≈ 7.83756e-5 in cards) | CSV + M1 + M2 | high | global | impact ionisation prefactor — frozen by Sebas |
| u0 | 0.048317 | M1, M2 | high | global | low-field mobility |
| vsatn | 1.0223e5 m/s | M1, M2 (`.param`) | high | global | saturation velocity |
| dvt0/1/2 | 1.9758 / 0.46322 / −0.035558 | M1, M2 | high | global | SCE coefficients |
| voff | −0.1368 V | M1, M2 | high | global | offset voltage |
| L_n (channel) | 0.18 µm | `.asc` `.param Ln=0.18u` | medium | global | M1 length; M2 uses Ln×10=1.8 µm |
| W_n | 0.36 µm | `.asc` | medium | global | both transistors |
| CBpar | 1 fF | `.asc` | medium | global | body-node parasitic cap (drives ramp-rate response — §6) |
| **M1-specific (DNWFB, top transistor)** | | | | | |
| etab (M1 card default) | +1.8 | M1 card | card default | M1 global | floating-body sign-flipped value |
| beta0 (M1 card default) | 19 | M1 card | card default | M1 global | overridden per-bias by CSV |
| k1 (M1 card default) | 0.53825 | M1 card | card default | M1 global | overridden per-branch by CSV |
| **M2-specific (bulk, bottom transistor)** | | | | | |
| etab | −0.086777 | M2 card | high | M2 global | standard small negative |
| beta0 | 18 | M2 card | high | M2 global | bulk reference |
| k1 | 0.63825 | M2 card | high | M2 global | bulk body-effect |
| **Per-(V_G1, V_G2) BSIM overrides (M1)** — drives the 3-branch fit | | | | | |
| K1 | 0.55825 / 0.53825 / 0.41825 | CSV + JSON | high (1% agreement) | per V_G1 branch only | constant across V_G2 in each branch |
| ETAB | red 0.80→1.10; blue 1.90→1.60; black 2.50 (flat) →2.10 | CSV + JSON | high | per-branch × V_G2 | floating-body weight, monotone in V_G2 |
| BETA0 | red 10.75→14.0; blue 19 (flat); black 20 (flat) | CSV + JSON | high | per-branch × V_G2 | flat-then-step structure |
| NFACTOR | red 12.15→6.25; blue 5.00→2.75; black 6.00→1.25 | CSV + JSON | high | per-branch × V_G2 | **dominant V_G2 dependence**, range ×10 |
| **Cell-level integration knobs** | | | | | |
| mbjt | 0.001 (V_G1=0.2) / 1.0 (V_G1=0.4, 0.6) | CSV | high | per V_G1 branch | parasitic-BJT area multiplier — 1000× step at V_G1≈0.3 V |
| trise | 9.04 (0.6) / ~11.6 (0.2) / 10.6–13.0 (0.4) | CSV | medium | per-bias | dynamic ramp parameter (units unspecified) |
| IS (cell BJT IS scale) | 5e-9 A | CSV | medium | global | matches parasiticBJT.is |
| area (cell) | 1e-6 | CSV | high | global | layout area used in BJT mbjt scaling |
| **Parasitic NPN (Gummel-Poon)** — D→Sint→B floating-body bipolar | | | | | |
| is | 5e-9 A | parasiticBJT.txt | high | global | saturation current |
| bf | 10000 | parasiticBJT.txt | high | global | forward β — very high (floating-body cell signature) |
| br | 100 | parasiticBJT.txt | high | global | reverse β |
| va | 100 V | parasiticBJT.txt | high | global | Early voltage |
| nc | 2 | parasiticBJT.txt | high | global | collector emission coef |
| ikr | 100 mA | parasiticBJT.txt | high | global | reverse knee |
| rc | 0.1 Ω | parasiticBJT.txt | high | global | collector resistance |
| vje | 0.7 V | parasiticBJT.txt | high | global | BE built-in |
| re | 0.1 Ω | parasiticBJT.txt | high | global | emitter resistance |
| cjc | 1 fF | parasiticBJT.txt | high | global | BC junction cap |
| cje | 0.7 fF | parasiticBJT.txt | high | global | BE junction cap |
| ne | 1.5 | parasiticBJT.txt | high | global | emitter ideality |
| tf / tr | 25 ps / 20 ps | parasiticBJT.txt | high | global | transit times |
| itf, vtf, xtf | 0.03 / 7 / 2 | parasiticBJT.txt | high | global | high-current rolloff |
| **Body diode** (pdiode) — Sint to Nwell/substrate | | | | | |
| level | 1 | pdiode.txt | high | global | generic SPICE diode |
| is | 5.3675e-7 A | pdiode.txt | high | global | sat current |
| n | 1.0535 | pdiode.txt | high | global | ideality |
| rs | 7.4e-8 Ω | pdiode.txt | high | global | series R |
| cj0 | 7.33e-4 F (per unit area) | pdiode.txt | high | global | zero-bias junction cap |
| vj | 0.21918 V | pdiode.txt | high | global | built-in potential (body junction, low) |
| m | 0.24097 | pdiode.txt | high | global | grading |
| bv | 11 V | pdiode.txt | high | global | breakdown |
| eg, xti | 1.11 eV, 6.5 | pdiode.txt | high | global | bandgap, sat-current temp exp |

---

## 5. image-2.png — content summary

Two-panel slide titled **"Dynamic response (ramp rate dependence)"**:

- **Left panel:** Measured I–V family ramped at multiple rates t_rise = 200 µs, 1 ms, 10 ms, 200 ms at V_D = 2 V, V_G2 = 0.45 V, V_G1 ∈ {0.5, 0.3 V}. Shows a **hysteresis "floating-bulk loop"** that opens up as the ramp slows — the bulk voltage tracks the drain ramp, and impact-ionisation carriers raise the floating body slowly. Annotation reads *"Filling and relaxation transients (slope S_FILL, S_REL) relate to speed of bulk voltage following the drain voltage increase (how fast impact ionisation generates carriers and increases the voltage of the floating body)"*.
- **Right panel:** Ramped simulation I–V family for the same biases — shows the model qualitatively reproduces the floating-bulk loop.
- **Right schematic:** The 2T cell with M1 (top, floating P-body) and M2 (bottom, V_G2 gate, V_NWELL=0 V, V_SINT=0 V). Highlights the BC junction parasitic cap and SR/Nwell-drain capacitance.
- **Annotations (red callouts):** "Parasitic capacitances play a crucial role on the effective time constant" and "SR and firing time experiments (3 through 7 in experiment list) are critical for fitting this dependence".

**Implication:** image-2.png is **not a parameter plot**. It is the rationale for why we need the transient experiments (3–7) and why **CBpar, tf/tr, cjc/cje, and trise** are first-order knobs for any dynamic match. No new numerical parameters from this image — but it nominates the parasitic capacitances and BJT transit times as the parameters Sebas considers most consequential for ramp-rate behaviour.

---

## 6. Recommended canonical set for SA4

1. **Freeze (do not refit):** all M1/M2 process globals (toxn, xj, ndep, vth0n, u0, vsatn, dvt*, voff, ua/ub/uc, w0, k2, k3, etc.), the parasiticBJT card in full, and pdiode in full. These are silicon-process and Sebas has not varied them across deliveries.
2. **Use CSV directly** as the authoritative per-bias override table for `K1, ETAB, BETA0, NFACTOR, mbjt` (on M1) and `trise` for transients. This supersedes the JSON (JSON is an independent re-reading of the same data, used only as cross-check — agreement is ≤1%).
3. **Branch structure is mandatory** for K1 (per V_G1), mbjt (per V_G1, step at 0.3 V), and BETA0 (per V_G1, step structure). Do NOT attempt a smooth 2-D polynomial fit for these. NFACTOR and ETAB can be smooth-in-V_G2 within each branch.
4. **Refit bound update:** widen NFACTOR upper bound to ≥15 (CSV peak 12.15; JSON note flags 15 as floor on bound).
5. **Floating-body signature** = (etab=+1.8 on M1, k1 reduced, beta0=19, mbjt=1 above V_G1≈0.3 V, CBpar=1 fF). This 5-tuple is the structural difference between M1 and M2 and is what gives the cell its NS-RAM behaviour.
6. **Ground-truth check set:** 33-bias `2vHCa-2 I-Vs@VG2 VG1=*/` measured CSVs (silicon, not fit) — use as held-out validation. CSV+JSON are derived from these via Sebas's hand-tuned fit.

---

## 7. Coverage / locked-gate status

- **JSON params catalogued:** 4 functional series (NFACTOR_M2, K1_M1, ETAB_M1, BETA0_M1) × 3 branches = ~12 per-branch curves. **All 4 cross-checked** against CSV at multiple points (table §3). **100% agreement** within ±0.05.
- **Card files:** M1, M2, parasiticBJT, pdiode, PTM130bulkNSRAM all read; process indicators (130 nm, toxn=4 nm, Sebas-tuned vth0n=0.541), comment-annotated parameters, and M1↔M2 diff documented.
- **image-2.png:** opened and described (§5); confirmed not a new parameter source — it is the transient rationale slide.
- **Unique params catalogued:** 21 process globals + 5 per-branch BSIM overrides + 16 BJT params + 12 pdiode params + 4 cell-level integration knobs ≈ **58 distinct parameters**.

Authoritative file for SA4 to consume programmatically: **`data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv`** (31-row per-bias table) + the four card files as-is (no edits).
