# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: R_deep_A_topology_compare.md (8319 chars) ===
```
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

```


=== FILE: R_deep_B_oracle_structural.md (5563 chars) ===
```
# R_deep_B — Oracle Synthesis: Structural vs Parametric? (O58_structural)

**Date**: 2026-05-13
**Packet**: `research_plan/oracle_queries/O58_structural/`
**Providers**: openai (gpt-5, 135s), gemini (2.5-pro, 68s), grok (4-latest, 59s)
**Wall**: ~4.4 min

---

## Per-question consensus / dissent

### Q1 — Structural vs parametric vs spurious?

| Oracle | Primary | Secondary |
|---|---|---|
| **gpt-5** | **C (spurious)** | B > A |
| **gemini** | **A (structural)** | C strong secondary |
| **grok** | **A (structural)** | C possible, B unlikely |

**Consensus**: it is **NOT pure (B) parametric**. All three reject "just sweep harder." 2/3 vote structural (A); gpt-5 inverts and says z304 was a spurious local optimum on wrong physics — but it agrees the v5b body-diode path is dead in KCL (which is itself a structural fact). So **the unanimous operational call is: there is at least one dead current path in v5b**, regardless of whether you call that "structural" or "parameter region with a dead branch".

**Dissent**: gpt-5 weights C > A because z304's Bf=9000 best-branch is nonphysical and v5b passes unit tests (no sign catastrophes). Gemini+grok weight A higher because the audit (`R3_pyport_audit.md`) explicitly flagged missing `body_pdiode_Rs` and the Js-invariance is bit-exact.

**Combined verdict**: Both are simultaneously true. z304 was a spurious optimum (fit-by-overfitting via nonphysical Bf and avalanche crutch) AND v5b has at least one structurally inert path (body diode). Removing the z304 crutches before fixing the v5b dead branch produced the regression.

### Q2 — Js invariance → which path dominates?

**Unanimous**: body p-n diode DC path is **inactive / negligible**. The dominant current paths are (in agreement):
1. Channel `Ids` (BSIM4)
2. Impact ionization `I_iii` (BSIM4 ALPHA0/BETA0 → body)
3. Parasitic BJT `I_bjt` (complementary firing source)

Diode role per Sebas: capacitive (Cb, transient time-constant), **not DC firing**. Adiode=22μm² and Cb=7fF make sense only in transient context.

Mechanism for inactivity (gemini + grok converge): missing `body_pdiode_Rs` series resistance means the diode branch is either clamped by the network or never reaches forward conduction in (Vd ∈ [0,2], V_G1 ∈ [0.2,0.6]) — body voltage floats low, diode stays off.

### Q3 — Why did "adding correct physics" regress V_G1=0.6?

**Unanimous mechanism**: *removing compensating errors*.

- z304 had three "crutches" giving it surplus DOF: (i) K1(VG2) instead of K1(VG1), (ii) ALPHA0 polynomial in (VG1,VG2), (iii) active avalanche/Chynoweth path, (iv) nonphysical Bf ≫ 50.
- v5b correctly removes (i)(ii)(iii) per Sebas's recipe and reframes BJT with Bf≈50.
- But v5b did NOT yet rewire the body voltage correctly (diode path dead). So the model lost its crutches before its real replacement mechanism became active → regression, especially at V_G1=0.6 where the avalanche crutch had been doing the most work.

### Q4 — Cheapest 2h discriminating experiment?

All three propose **path-liveness ablation**, differing in implementation:

| Oracle | Design |
|---|---|
| gpt-5 | Toggle 3 mechanisms (`iii_to_body_factor=0`, `mbjt=0`, `body_pdiode_to="off"`) at 3 bias corners. Gate: if A1 and A2 both move <0.1 dec → structural fault. |
| gemini | Single-cell `use_well_diode=True` with vnwell_Rs=1e8Ω vs current control. Gate: >1% rmse change → structural confirmed. |
| grok | 5×5 Bf×Js sweep on full v5b + add body_pdiode_Rs. Gate: any combo <1.0 dec → parametric; all ≥3.0 → structural. |

---

## Verdict

**Structural with parametric-amplification**. The v5b model has at least one dead KCL branch (body p-n diode, Js-invariant by bit-exact test). The previous z304 "success" was a spurious local optimum riding on now-removed crutches (overfit Bf, K1(VG2), avalanche). You cannot resolve this with BBO until the body-voltage path is electrically live.

**Order of operations**:
1. Make the body branch live (add `body_pdiode_Rs` OR re-enable the existing `vnwell_Rs` path).
2. Verify Js sweep now produces non-identical residuals (positive control on structural fix).
3. Then BBO over (Bf, K1_LUT_scale, mbjt_step_threshold, BETA0_scale).

## Recommended cheapest 2h experiment (synthesized)

**Two-stage liveness ablation, ≤2h on daedalus**:

**Stage 1 (≤30 min) — Liveness positive control**
- Single cell V_G1=0.6, V_G2=0.0, recipe = v5b.
- Variant A (control): current v5b.
- Variant B: enable `use_well_diode=True` with `vnwell_Rs=1e8 Ω` (gemini's path — uses already-wired infrastructure).
- Variant C: kill BSIM impact-ionization (`iii_to_body_factor=0`).
- Variant D: kill BJT (`mbjt=0`).
- **PASS structural confirmed if**: B differs from A by ≥1% RMSE AND (C or D) shifts RMSE by ≥0.5 dec.
- **FAIL (parametric only) if**: A=B bit-exact and C,D both move <0.1 dec → no path is live; deeper structural problem than body diode.

**Stage 2 (≤90 min) — Mini BBO conditional on Stage-1 result**
- If structural confirmed: fix `body_pdiode_Rs` properly, then run 5×5 (Bf, K1_LUT_scale) on 3 representative cells (9 fits × ~10min ≈ 90 min).
- Pre-registered success: any combo <1.5 dec at V_G1=0.6.

If Stage 2 still fails to recover ≤1.5 dec, the structural flaw extends beyond the body diode (likely BJT polarity / iii→body sign / Vb node consumption).

---

## Files
- `research_plan/oracle_queries/O58_structural/prompt.md`
- `research_plan/oracle_queries/O58_structural/openai_response.md`
- `research_plan/oracle_queries/O58_structural/gemini_response.md`
- `research_plan/oracle_queries/O58_structural/grok_response.md`

```


=== FILE: _normalised/cell_asc.txt (1419 chars) ===
```
Version 4
SHEET 1 3052 680
WIRE 800 0 512 0
WIRE 800 64 800 0
WIRE 816 64 800 64
WIRE 848 64 816 64
WIRE 512 112 512 0
WIRE 800 112 800 64
WIRE 608 160 512 160
WIRE 640 160 608 160
WIRE 688 160 640 160
WIRE 704 160 688 160
WIRE 736 160 704 160
WIRE 752 160 736 160
WIRE 432 192 384 192
WIRE 464 192 432 192
WIRE 512 240 512 208
WIRE 800 240 800 208
WIRE 800 240 512 240
WIRE 608 272 608 160
WIRE 800 272 800 240
WIRE 704 288 704 160
WIRE 624 320 608 320
WIRE 544 352 496 352
WIRE 560 352 544 352
WIRE 624 368 624 320
WIRE 624 368 608 368
WIRE 800 400 800 272
WIRE 608 416 608 368
WIRE 704 416 704 352
FLAG 640 160 B
FLAG 800 272 Sint
FLAG 816 64 D
FLAG 704 416 0
FLAG 608 416 0
FLAG 432 192 G
FLAG 544 352 G2
FLAG 384 192 G
IOPIN 384 192 In
FLAG 496 352 G2
IOPIN 496 352 In
FLAG 848 64 Din
IOPIN 848 64 In
FLAG 800 400 S
IOPIN 800 400 Out
FLAG 688 160 B
IOPIN 688 160 Out
SYMBOL npn 736 112 R0
SYMATTR InstName Q1
SYMATTR Value parasiticBJT
SYMATTR Value2 area=1u
SYMBOL nmos4 464 112 R0
SYMATTR InstName M1
SYMATTR Value2 l='Ln' w='Wn' m=1
SYMBOL cap 688 288 R0
WINDOW 3 22 49 Left 2
SYMATTR Value 'CBpar'
SYMATTR InstName C1
SYMATTR SpiceLine Rser=1m
SYMBOL nmos4 560 272 R0
SYMATTR InstName M2
SYMATTR Value2 l='Ln*10' w='Wn' m=1
TEXT 552 24 Left 2 !.param Ln=0.18u\n.param Wn=0.36u\n.param CBpar=1f
TEXT 520 -64 Left 2 !.inc PTM130bulkNSRAM.txt
TEXT 520 -40 Left 2 !.inc parasiticBJT.txt
TEXT 310 478 Left 2 !.op 0

```


=== FILE: nsram_cell_2T_head.py (4276 chars) ===
```python
"""nsram_cell_2T — Differentiable 2T NS-RAM cell with proper topology.

Replaces the 1T proxy in `nsram_cell.py` (which collapses VG2 into a
``vth0_eff = vth0 + gamma·VG2`` shift) with the FULL 2T topology faithful
to Sebas's schematic ``data/sebas_2026_04_22/2tnsram_simple.asc``::

        D ──┬─────────────┬── (drain pin)
            │             │
          M1.D          Q1.C
            │             │
   VG1 → M1.G           Q1.B ── B  (floating body, shared by M1 & M2)
            │             │
          M1.S ── Sint ── Q1.E
                    │
                  M2.D
                    │
   VG2 → M2.G       │
                    │
                  M2.S ── 0  (ground)

Two NMOS (M1 short, M2 long) share floating body B. The internal node
Sint is the M1 source / M2 drain / Q1 emitter. Two unknown internal
voltages (Vsint, Vb) are solved by Newton-Raphson at each (Vd, VG1, VG2)
bias point so Sint-KCL = 0 and Body-KCL = 0.

Newton residuals (currents INTO each node):

    R_Sint(Vsint, Vb) =
        + Ids_M1(VG1−Vsint, Vd−Vsint, Vb−Vsint)            # M1 source ejects into Sint
        − Ids_M2(VG2,         Vsint,    Vb)                 # M2 drain absorbs from Sint
        + Ie_Q1(Vb−Vsint, Vb−Vd)                            # BJT emitter ejects into Sint
        + Ibs_diode_M1(Vb−Vsint)                            # forward body→Sint diode of M1
        − Ibd_diode_M2(Vb)                                  # forward body→drain(=Sint) of M2 leaves Sint

    R_B(Vsint, Vb) =
        + Iii_M1 + Iii_M2                                   # impact-ion holes → body
        + Igidl_M1 + Igisl_M1 + Igidl_M2 + Igisl_M2         # BTBT
        + Igb_M1 + Igb_M2                                   # gate→body tunnel
        − Ibd_diode_M1(Vb−Vd) − Ibs_diode_M1(Vb−Vsint)      # M1 junction leaks LEAVE body
        − Ibd_diode_M2(Vb)    − Ibs_diode_M2(Vb)            # M2 junction leaks LEAVE body
        − Ib_Q1(Vb−Vsint, Vb−Vd)                            # BJT base current leaves B

Drain terminal current at the D pin (positive into device):
    Id = Ids_M1 + Ic_Q1 + Igidl_drain_M1 + Ibd_diode_M1

VG2 is now a *real* gate to M2 (not a proxy threshold shift); body-effect
on M1 enters naturally via Vbs_M1 = Vb − Vsint.

Differentiability: simplest correct path. Newton iterations live INSIDE
autograd (no implicit-function-theorem trick yet). Each iteration is a
single forward of the full BSIM4 stack (~30 calls per bias point worst
case, double precision). For 33×~10 sweep points that's still tractable.

WARNING: do NOT add arbitrary clipping to "fix" Newton divergence — that
was the v4 mistake. Diagnose with `verbose=True`.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch

from .bjt import GummelPoonNPN, compute_bjt
from .dc import compute_dc
from .diode import compute_body_diodes
from .geometry import Geometry
from .leak import compute_iimpact, compute_igidl_gisl, compute_igb
from .model_card import BSIM4Model
from .temp import compute_size_dep, SizeDependParam


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class NSRAMCell2TConfig:
    """Static config for a 2T NS-RAM cell.

    Geometry + toggles + Newton solver knobs. Two SizeDependParam objects
    (one per MOSFET) are computed lazily.
    """
    Ln: float = 180e-9                  # M1 channel length [m]
    Wn: float = 360e-9                  # both channels' width [m]
    M2_length_factor: float = 10.0      # M2 length = Ln * factor (Sebas: 10x)
    Cbody: float = 1e-15                # body cap [F] (transient only; from CBpar)
    T_C: float = 27.0                   # operating temperature

    # Junction geometry per MOSFET. None → auto W·L / 2(W+L).
    As_M1: Optional[float] = None
    Ad_M1: Optional[float] = None
    Ps_M1: Optional[float] = None
    Pd_M1: Optional[float] = None
    As_M2: Optional[float] = None
    Ad_M2: Optional[float] = None
    Ps_M2: Optional[float] = None
    Pd_M2: Optional[float] = None

    # Toggle physics

```


=== FILE: z329_summary.json (11086 chars) ===
```json
{
  "script": "z329_iii_vsint_map",
  "device": "cpu",
  "V_d": 2.0,
  "n_vsint_steps": 50,
  "iii_floor": 1e-25,
  "alpha0_const": 7.842e-05,
  "n_biases_total": 33,
  "n_biases_valid": 25,
  "n_biases_non_trivial": 25,
  "median_Vsint_transition": 0.6530612244897959,
  "mean_Vsint_transition": 0.6612244897959184,
  "min_Vsint_transition": 0.5714285714285714,
  "max_Vsint_transition": 0.6938775510204082,
  "gates": {
    "INFRA_n_nontrivial_ge_30": false,
    "PASS_median_lt_0p7_Vd": true
  },
  "pass_threshold_V": 1.4,
  "per_bias": [
    {
      "idx": 0,
      "VG1": 0.2,
      "VG2": -0.2,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 1.6153635326071538e-13,
      "Iii_at_Vsint0": 1.6153635326071538e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 4.0079741558001185e-14
    },
    {
      "idx": 1,
      "VG1": 0.2,
      "VG2": -0.15,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 1.4216087107046206e-13,
      "Iii_at_Vsint0": 1.4216087107046206e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 4.0079741558001185e-14
    },
    {
      "idx": 2,
      "VG1": 0.2,
      "VG2": -0.1,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6122448979591836,
      "Iii_max": 1.251093816070901e-13,
      "Iii_at_Vsint0": 1.251093816070901e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 4.0079741558001185e-14
    },
    {
      "idx": 3,
      "VG1": 0.2,
      "VG2": -0.05,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6122448979591836,
      "Iii_max": 1.1010313350113341e-13,
      "Iii_at_Vsint0": 1.1010313350113341e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 4.0079741558001185e-14
    },
    {
      "idx": 4,
      "VG1": 0.2,
      "VG2": 0.0,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6122448979591836,
      "Iii_max": 6.604483892649871e-14,
      "Iii_at_Vsint0": 6.604483892649871e-14,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 4.0079741558001185e-14
    },
    {
      "idx": 5,
      "VG1": 0.2,
      "VG2": 0.05,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6122448979591836,
      "Iii_max": 3.961668128893227e-14,
      "Iii_at_Vsint0": 3.961668128893227e-14,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 4.0079741558001185e-14
    },
    {
      "idx": 6,
      "VG1": 0.2,
      "VG2": 0.1,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.5714285714285714,
      "Iii_max": 3.06829911492676e-14,
      "Iii_at_Vsint0": 3.06829911492676e-14,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 4.0079741558001185e-14
    },
    {
      "idx": 7,
      "VG1": 0.4,
      "VG2": -0.2,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 8,
      "VG1": 0.4,
      "VG2": -0.15,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 9,
      "VG1": 0.4,
      "VG2": -0.1,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 10,
      "VG1": 0.4,
      "VG2": -0.05,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 11,
      "VG1": 0.4,
      "VG2": 0.0,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 9.256353185233838e-13,
      "Iii_at_Vsint0": 9.256353185233838e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 1.5571801663425606e-11
    },
    {
      "idx": 12,
      "VG1": 0.4,
      "VG2": 0.05,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 9.256353185233838e-13,
      "Iii_at_Vsint0": 9.256353185233838e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 1.5571801663425606e-11
    },
    {
      "idx": 13,
      "VG1": 0.4,
      "VG2": 0.1,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 9.256353185233838e-13,
      "Iii_at_Vsint0": 9.256353185233838e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 1.5571801663425606e-11
    },
    {
      "idx": 14,
      "VG1": 0.4,
      "VG2": 0.15,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 9.256353185233838e-13,
      "Iii_at_Vsint0": 9.256353185233838e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 1.5571801663425606e-11
    },
    {
      "idx": 15,
      "VG1": 0.4,
      "VG2": 0.2,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 9.256353185233838e-13,
      "Iii_at_Vsint0": 9.256353185233838e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 1.5571801663425606e-11
    },
    {
      "idx": 16,
      "VG1": 0.4,
      "VG2": 0.25,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 9.256353185233838e-13,
      "Iii_at_Vsint0": 9.256353185233838e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 1.5571801663425606e-11
    },
    {
      "idx": 17,
      "VG1": 0.4,
      "VG2": 0.3,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6530612244897959,
      "Iii_max": 9.256353185233838e-13,
      "Iii_at_Vsint0": 9.256353185233838e-13,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 1.5571801663425606e-11
    },
    {
      "idx": 18,
      "VG1": 0.6,
      "VG2": -0.2,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 19,
      "VG1": 0.6,
      "VG2": -0.15,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 20,
      "VG1": 0.6,
      "VG2": -0.1,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 21,
      "VG1": 0.6,
      "VG2": -0.05,
      "V_d": 2.0,
      "valid": false,
      "reason": "K1 nan",
      "Vsint_transition": null,
      "Iii_max": null,
      "Iii_at_Vsint0": null
    },
    {
      "idx": 22,
      "VG1": 0.6,
      "VG2": 0.0,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 23,
      "VG1": 0.6,
      "VG2": 0.05,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 24,
      "VG1": 0.6,
      "VG2": 0.1,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 25,
      "VG1": 0.6,
      "VG2": 0.15,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 26,
      "VG1": 0.6,
      "VG2": 0.2,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 27,
      "VG1": 0.6,
      "VG2": 0.25,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 28,
      "VG1": 0.6,
      "VG2": 0.3,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 29,
      "VG1": 0.6,
      "VG2": 0.35,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 30,
      "VG1": 0.6,
      "VG2": 0.4,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 31,
      "VG1": 0.6,
      "VG2": 0.45,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    },
    {
      "idx": 32,
      "VG1": 0.6,
      "VG2": 0.5,
      "V_d": 2.0,
      "valid": true,
      "non_trivial": true,
      "Vsint_transition": 0.6938775510204082,
      "Iii_max": 1.1832188793524552e-11,
      "Iii_at_Vsint0": 1.1832188793524552e-11,
      "Iii_at_Vsint_Vd": 0.0,
      "Ids_at_Vsint0": 3.3217691525053563e-10
    }
  ],
  "heatmap_png": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z329_iii_vsint_map/iii_heatmap.png",
  "hist_png": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z329_iii_vsint_map/vsint_transition_hist.png",
  "elapsed_s": 5.9402241706848145
}
```


=== FILE: z330_summary.json (567 chars) ===
```json
{
  "bias": {
    "VG1": 0.6,
    "VG2": 0.2,
    "Vd": 2.0
  },
  "pyport_ref": {
    "Vsint": 1.867,
    "Vb": 2.0
  },
  "pyport_recomputed": {
    "Id": 0.5834617425274572,
    "Iii_in": 5.34232933125968e-21,
    "Ileak_out": 0.00023893090248603988,
    "Vsint": 1.0,
    "converged": false
  },
  "ngspice": {
    "rc": 0,
    "Vsint": 0.3823582,
    "Vb": 0.2673754,
    "Vd": 2.0,
    "Id": 3.93432e-11
  },
  "diff_vsint_abs_V": 1.4846418,
  "gate_pass_pyport_bug_confirmed": true,
  "verdict": "PASS \u2014 pyport solver bug CONFIRMED (>0.5V disagreement)"
}
```


=== FILE: z331_summary.json (1922 chars) ===
```json
{
  "script": "z331_snapback_graph",
  "VG1": 0.4,
  "VG2_LIST": [
    0.0,
    0.2,
    0.4
  ],
  "n_vd": 80,
  "vd_range": [
    0.0,
    4.0
  ],
  "per_vg2": [
    {
      "vg2": 0.0,
      "row_vg2": 0.0,
      "model_peak_Vd": 2.6835443037974684,
      "model_peak_Id": 5.490882785572069e-14,
      "measured_peak_Vd": 1.95024,
      "measured_peak_Id": 4.11312e-06,
      "measured_file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.4(1)_03-37-28PM.csv",
      "measured_peak_at_sweep_edge": true,
      "infra_knee_in_1p5_3": true,
      "pass_peak_within_0p5V": false,
      "log_rmse": 6.076859625659037,
      "ambitious_log_rmse_lt_0p5": false
    },
    {
      "vg2": 0.2,
      "row_vg2": 0.2,
      "model_peak_Vd": 2.5316455696202533,
      "model_peak_Id": 1.2899032672129466e-12,
      "measured_peak_Vd": 1.95036,
      "measured_peak_Id": 4.10957e-06,
      "measured_file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
      "measured_peak_at_sweep_edge": true,
      "infra_knee_in_1p5_3": true,
      "pass_peak_within_0p5V": false,
      "log_rmse": 4.162084333966397,
      "ambitious_log_rmse_lt_0p5": false
    },
    {
      "vg2": 0.4,
      "row_vg2": 0.3,
      "model_peak_Vd": 2.0253164556962027,
      "model_peak_Id": 1.4293043161082904e-11,
      "measured_peak_Vd": 2.15032,
      "measured_peak_Id": 7.60879e-06,
      "measured_file": "StandardIV_HH_2vHCa-2_VG2=0.3_VG=0.4(1)_03-40-47PM.csv",
      "measured_peak_at_sweep_edge": true,
      "infra_knee_in_1p5_3": true,
      "pass_peak_within_0p5V": true,
      "log_rmse": 3.024942984841462,
      "ambitious_log_rmse_lt_0p5": false
    }
  ],
  "png": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z331_snapback_graph/z331_snapback_graph.png",
  "gate": {
    "INFRA_all_knee_in_range": true,
    "PASS_all_peak_within_0p5V": false,
    "AMBITIOUS_all_log_rmse_lt_0p5": false
  },
  "elapsed_s": 16.31
}
```
