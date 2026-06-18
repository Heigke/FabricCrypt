# R_deep_A — Topology Comparison: LTSpice `2tnsram_simple.asc` vs pyport_v5 `_residuals`

Date: 2026-05-13
Source files:
- LTSpice: `data/sebas_2026_04_22/2tnsram_simple.asc` + `parasiticBJT.txt` + `PTM130bulkNSRAM.txt`
- pyport: `nsram/nsram/bsim4_port/nsram_cell_2T.py` (post R-3+R-4 wiring)

---

## 1. Nodes

| LTSpice .asc        | pyport `_residuals` arg | Notes                                 |
|---------------------|-------------------------|---------------------------------------|
| `Din` / `D`         | `Vd`                    | external pin (input)                  |
| `S`                 | hard-coded `0` (GND)    | external output pin = ground          |
| `Sint`              | `Vsint` (solved)        | floating internal node                |
| `B`                 | `Vb` (solved)           | floating bulk                         |
| `G`                 | `VG1`                   | M1 gate                               |
| `G2`                | `VG2`                   | M2 gate                               |
| (no `Nwell` flag)   | `cfg.vnwell` (param)    | LTSpice does NOT show a Nwell node    |
| GND (flag 0)        | 0                       |                                       |

**LTSpice node count = 6** (D, Sint, B, G, G2, GND). **No Nwell node.**
**pyport node count = 6 solved + Nwell-as-parameter** (one extra virtual node).

## 2. Devices

| LTSpice (4 devices)                                                | pyport (5+ devices)                                                                            |
|---|---|
| **M1** `nmos4` PTM130bulkNSRAM, L=Ln, W=Wn, D=D, G=G, S=Sint, B=B  | M1 BSIM4 D=Vd G=VG1 S=Vsint B=Vb ✓                                                              |
| **M2** `nmos4` PTM130bulkNSRAM, L=10·Ln, W=Wn — D=Sint, G=G2, S=GND, **B = (left unconnected → GND)** | M2 BSIM4 D=Vsint G=VG2 S=0 B=`zero if cfg.m2_body_gnd else Vb` ✓ |
| **Q1** `parasiticBJT` NPN, **C=D, B=B, E=Sint** (per wire trace: pin (752,112) on D-rail; pin (752,208) on Sint-rail; pin (~736,160) on B-rail) | NPN compute_bjt with **Vbe=Vb (E=GND), Vbc=Vb−Vd**  — **EMITTER = GND, NOT Sint** |
| **C1** cap `CBpar` = 1 fF Rser=1m, from **B → GND** (top pin 704,288 ↔ B-net y=160; bottom pin 704,352 → GND flag at 704,416) | NOT in DC residuals (caps inactive in `.op`); `Cbody` only used in transient |
| (none — no pdiode/well-diode device in netlist)                    | **vnwell well-diode** (`use_well_diode`) + **body_pdiode** (`body_pdiode_to`) with optional series-Rs (R-4) + optional TAT current — none of which exists in LTSpice |
| (none)                                                             | iii_gain bookkeeping, lateral collector (Ic_lat = Bf·Ib_lat), avalanche multiplier, local-base inner Newton |

**LTSpice device count = 4** (M1, M2, Q1, C1).
**pyport effective device count = 4 + (1 vnwell + 1 body_pdiode + 1 TAT) = up to 7** in DC.

## 3. Numbered Discrepancies (ordered by likelihood of causing v5b regression)

### D1. **Q1 emitter wired to GND, not Sint**  [HIGH]
- LTSpice wire trace: 800-col gap between y=112 (D-rail) and y=208 (Sint-rail) is exactly where Q1 sits at (736,112). NPN R0 pins land C@(752,112)→D-net, E@(752,208)→**Sint-net**, B@(~736,160)→B-net.
- pyport (line 510-515): `Vbe = Vb` with comment "emitter = ground (legacy F1.v2 path)" — explicit deviation justified by "A.1.i finding" claim.
- Consequence: With E=GND, Q1 turns on at Vb~0.6 V drawing current from D→GND, completely bypassing Sint. With E=Sint (true LTSpice), Vbe = Vb − Vsint, Q1 only fires when Vb leads Vsint — fundamentally different snapback dynamics. **R_Sint also missing a +Ie_Q1 term** (line 535: no BJT current touches Sint at all, but in LTSpice Q1 sources Ie INTO Sint).

### D2. **Extra Nwell-coupled diodes that do not exist in netlist**  [HIGH]
- LTSpice has **zero** explicit diode devices and **no Nwell node**. The N-well/p-substrate junction is implicit in BSIM4 `dnwell` parameters of PTM130bulkNSRAM (handled inside the MOSFET model itself).
- pyport: `use_well_diode=True` (default) injects `I_well_body = mbjt · Js·A·(exp(...)−1)` between a phantom `vnwell` parameter and Vb. Plus `body_pdiode_to="vnwell"` adds *another* parallel diode at the same junction. With Bf=50 and v5b R-4 series-R the body is now pinned to vnwell, killing snapback.
- This explains why "adding physical elements made it worse": LTSpice models the well junction implicitly once, pyport models it explicitly twice (well_diode + body_pdiode) AND adds the BSIM4 internal one. Triple-count.

### D3. **CBpar (1 fF B→GND) missing in DC residuals (silent in `.op`, but flagged because v5b enables transient elsewhere)**  [MED]
- LTSpice C1: B → GND, 1 fF. Inactive in `.op 0` so does not affect DC.
- pyport: `Cbody` parameter exists but is not referenced in `_residuals` at all. **Polarity check**: any transient path elsewhere must use B→GND, not B→Sint.

### D4. **mbjt scaling has no physical analog**  [MED]
- pyport multiplies `I_well_body *= cfg.vnwell_mbjt`. There is no per-bias scaling factor for the well diode in LTSpice (the MOSFET's BSIM4 internal junction is sized by area only).
- This was a fitting kludge to fight D2's overcounting and breaks when v5b switches to Sebas's published Bf.

### D5. **`m2_body_gnd` defaults / branch divergence**  [MED]
- LTSpice: M2.B is **floating-unconnected** in the symbol → LTSpice defaults to `0` (GND) — pyport gets this right when `m2_body_gnd=True`.
- But the residual has two large code branches (`m2_body_gnd` vs not). The "not" branch subtracts `m2["Ibs"]+m2["Ibd"]` from Vb (treating M2.B=Vb) which contradicts the LTSpice schematic.
- Confirm default is `m2_body_gnd=True`; if any v5 caller passes False, body is double-leaked.

### D6. **Series-R on body_pdiode = 1e10 Ω (R-4 default)**  [LOW]
- Without any physical analog. Effectively makes body_pdiode behave as resistor (since exp current dwarfs 1e10 Ω drop only at very high V). LTSpice has no such resistor.
- LOW likelihood as primary culprit (large Rs ≈ disabling it), but interacts with D2 unpredictably.

### D7. **Avalanche multiplier removed per R-1b** [LOW]
- LTSpice PTM130bulkNSRAM uses BSIM4 Iii (impact-ionization) for avalanche. pyport `use_lateral_collector=False` default. Consistent with R-1b mail. No discrepancy in current default.

### D8. **iii_gain inflation in body KCL** [LOW]
- pyport inflates Iii by `iii_gain` (default >1 with sigmoid). LTSpice uses raw BSIM4 Iii once. This is a model-tuning, not topology, divergence.

### D9. **NPN Bf**: parasiticBJT.txt has **Bf=10000** [HIGH context, not strictly a residuals bug]
- The instruction text says "Sebas's published Bf=50" but the model card file shows `bf=10000`. If v5b is using Bf=50 vs the file's 10000, the BJT is 200× weaker — but with E=GND (D1 wrong), even Bf=10000 produces the wrong qualitative behavior.

## 4. Top 3 Fixes (in order)

1. **Fix D1: Wire Q1.E to Sint.**
   - `nsram_cell_2T.py:514-519` — change `Vbe = Vb` → `Vbe = Vb - Vsint` and `Vbc = Vb - Vd` stays.
   - `nsram_cell_2T.py:535` (`R_Sint`) — add **`+ Ie_Q1`** (emitter current into Sint; sign: `Ie_Q1` from `compute_bjt` is current leaving emitter, so flows INTO Sint when BJT is forward).  Verify sign with bjt.py.
   - Expected effect: snapback regime changes from "Vb-only trigger" to "Vb-leads-Vsint trigger" matching LTSpice physics.

2. **Fix D2: Disable extraneous well/body diodes by default.**
   - `nsram_cell_2T.py:117` set `use_well_diode: bool = False`.
   - `nsram_cell_2T.py:162` set `body_pdiode_to: str = "off"`.
   - Rationale: LTSpice models the N-well junction implicitly inside PTM130bulkNSRAM's BSIM4 (`dnwell`/source-bulk diode). Explicit diodes triple-count.
   - If a "vnwell knob" is required for the V_Nwell sweep experiments, expose it ONLY through BSIM4 `nstype`/`dnwell` model parameters, not as an additional diode device.

3. **Fix D9 (sanity): Use parasiticBJT.txt Bf=10000.**
   - Wherever `GummelPoonNPN` is constructed for the 2T cell, source `Bf` from `data/sebas_2026_04_22/parasiticBJT.txt` (`bf=10000`), not from a separate "published" value of 50/100.
   - Combined with Fix 1, will restore the strong reverse-Early/snapback that LTSpice produces.

## 5. Gate Status

≥3 structural discrepancies identified at HIGH likelihood (D1, D2, D9). Gate **OPEN**.
