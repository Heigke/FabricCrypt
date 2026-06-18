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
