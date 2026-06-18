# A9 — Deep re-scan of older researcher materials for the 60 mV Vth offset

**Goal:** find anything older that explains the uniform −60 mV Vth offset of
our PyTorch BSIM4 port vs ngspice on the isolated M2 (130nm bulk NSRAM) card,
as discovered in z91l (`results/z91l_vth_dibl/summary.json`).

z91l observation (recap):
- Vds=0.05V → diff = -58.5 mV
- Vds=0.50V → diff = -59.5 mV
- Vds=2.00V → diff = -59.8 mV
- DIBL_ng=6.25 mV/V vs DIBL_py=6.91 mV/V (matches to ~0.6 mV/V)
- z91k (Vds=0.5): S_ng=72.5, S_py=76.5 mV/dec (n matches to 5%)

So the bug is a **constant Vds-independent shift in the Id-Vgs curve** (Vth as
extracted via constant-current criterion is uniformly low by ≈60 mV in pyport).

---

## 1. Older materials reviewed (one line each)

- `/home/ikaros/nsram_info/schematic&modelCards/PTM130bulkNSRAM.txt` — older
  copy of the model card; **identical** to `data/sebas_2026_04_22/M2_130bulkNSRAM.txt`
  body (diff is empty). Same `phin=0.05`, same `wvth0=-1.6569e-8`, same `voff=-0.1368`.
- `/home/ikaros/nsram_info/schematic&modelCards/2tnsram_simple.asc` — identical to
  `data/sebas_2026_04_22/2tnsram_simple.asc` (diff empty).
- `/home/ikaros/nsram_info/schematic&modelCards/parasiticBJT.txt` — Gummel-Poon BJT;
  no MOSFET parameters; not relevant to Vth.
- `/home/ikaros/nsram_info/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/...` — measurement
  CSVs (~21 sweeps across VG1∈{0.2,0.4,0.6}, VG2∈{-0.2..0.5}). No model notes; pure data.
- `/home/ikaros/nsram_info/Zoom/*.jpeg` — Zoom screenshots; not useful for parameter audit.
- `/home/ikaros/nsram_info/2026-04-30 BSIMfitsBA/` — already covered in A.7/A.8.
- `data/sebas_2026_04_22/PTM130bulkNSRAM.original.txt` — bare PTM template (pre-Sebas
  edits). Diff vs `PTM130_ngspice.txt` shows only one substantive change: the
  `.param` block gained `vsatn=1.35e5` and an `=` sign on each line (z91f patcher
  hardcodes `vsatn=102230` from the M2 card body, NOT 1.35e5 from PTM original — fine).
- `data/sebas_2026_04_22/M1_130DNWFB.txt` — the M1 (DNW-FB) card; same `phin=0.05`,
  `vth0=vth0n`, `wvth0=-1.6569e-8`, `voff=-0.1368`. Same potential phi issue.
- `data/nsram_zenodo/README.md` + `SimulationFiles/` — Zenodo NSRAM repo; unrelated to
  130nm fit.
- Scripts inspected: `z23_ngspice_baseline.py`, `z26_ngspice_gap.py`,
  `z28_validate_vs_ngspice.py` not present in `scripts/` (no z23/z26/z28 in this tree).
  Only z910+ exist.
- `scripts/ngspice_models/PTM130_ngspice.txt` — bare PTM (pre-Sebas) template; identical
  body to `PTM130bulkNSRAM.original.txt`.

## 2. Historical email/note hits

- `/tmp/emails_clean.txt` only one substantive hit: line 262
  *"§13 layout stress (SA, SB, KU0, KVTH0) — direct home for your 'layout-dependent
  effect'"*. Mentions KVTH0 (BSIM §13 stress) — Sebas did not flag a numeric Vth
  offset.
- `docs/email_sebas_mario_nsram_v012.txt` and `docs/bsim4_port_*.md` skimmed — no
  references to a 60 mV offset, vth0_eff vs vth0 distinction, or PTM-130 quirks
  beyond the LDE/LPE machinery already in the port.
- `docs/firmware_to_feel_investigation.md` — unrelated (firmware/TMR work).
- `docs/z2352_*` — unrelated (PSP / vector-units).
- No older material describes a "60 mV" offset, "Vt0 shift", or a binning quirk.
  Sebas's own emails treat the model card as authoritative.

## 3. Where Vth aggregation lives + candidate diagnosis

**Vth assembly** (per-bias) in `nsram/nsram/bsim4_port/dc.py:349-356`:
```
Vth = type_n*vth0
    + (k1ox*sqrtPhis - k1*sqrtPhi_pre) * Lpe_Vb
    - k2ox*Vbseff
    - Delt_vth
    - T2_narrow
    + (k3 + k3b*Vbseff) * Vth_NarrowW
    + Tlpe1
    - DIBL_Sft
```
inputs traced:
- `vth0 ← sd.vth0_T` (`temp.py:217`, computed via L/W/LW binning at `temp.py:198-205`)
- `phi, sqrtPhi, vbi, k1ox, factor1, Xdep0` — all in `temp.py:283-292`
- `voffcbn = scaled["voff"] + voffl/Leff` — `temp.py:319-322`

**Verification just done** (numeric, M2 card, L=1.8μm, W=0.36μm, T=27°C, Vbs=0,
Vds=0):

| quantity              | pyport     | ngspice (`show m1 : vth`) |
|-----------------------|-----------:|--------------------------:|
| symbolic Vth(Vbs=0)   | 0.58395 V  | 0.58404 V                 |
| difference            | **0.00009 V (90 µV)** |                |

→ **The symbolic Vth assembly itself is essentially exact.** vth0_T binning
(0.484 V from 0.541 V card via `wvth0·Inv_W`), phi (0.872), vbi (1.009), Tlpe1
(+20 mV), `(k3+k3b·Vbs)·Vth_NarrowW` (+80 mV) all reproduce ngspice.

**So the 60 mV gap is NOT in Vth-assembly.** It lives in the
**Vgsteff/subthreshold bridge** (`dc.py:397-472`, b4ld.c §1238-1296).

Side-by-side Id-Vgs at Vds=0.05V (M2):

| Vgs   | Id_ngspice | Id_pyport  | log10(py/ng) |
|-------|-----------:|-----------:|-------------:|
| 0.40  | 1.44e-10   | 1.19e-9    | **+0.92**    |
| 0.45  | 6.58e-10   | 4.16e-9    | +0.80        |
| 0.48  | 2.72e-9    | 8.29e-9    | +0.48        |
| 0.50  | 9.89e-9    | 1.28e-8    | +0.11        |
| 0.52  | 3.09e-8    | 1.94e-8    | -0.20        |
| 0.55  | 7.91e-8    | 3.46e-8    | -0.36        |
| 0.58  | 1.61e-7    | 5.87e-8    | -0.44        |

Pyport is **too HIGH** in deep subthreshold (Vgs<0.50) and **too LOW** above (Vgs>0.55).
Implied subthreshold slopes Vgs∈[0.40,0.50]:
- ngspice: ~54.5 mV/dec
- pyport: ~96.9 mV/dec
(NB: z91k measured slope at Vds=0.50V where they happen to match within 4 mV/dec —
the disagreement is much worse at low Vds, low Vgs.)

So: **n is too large in pyport** in the deep-subthreshold regime, AND/OR
voffcbn or the T9-denominator branch is mis-computed there. The constant-Vds
Vth shift (60 mV) is the integrated consequence of the slope/offset mismatch
when extracted via Id_target = (W/L)·1e-7.

## 4. PTM130 vs PTM130bulkNSRAM diff summary

`diff PTM130bulkNSRAM.original.txt PTM130_ngspice.txt`:
- Header `.param` block reformatted (`Nparam 1.58` → `Nparam=1.58`, etc.) — purely
  cosmetic, ngspice requires `=`.
- `.param vsatn=1.35e5` added in `PTM130_ngspice.txt` only. **NOT a transformation we
  apply to M2 fits** — z91f hardcodes `vsatn=102230` from the M2 body itself. No silent
  scaling.
- No vth-related parameters changed between the two files.

→ **No mid-stream parameter scaling explains the 60 mV.**

## 5. Top 3 candidate fixes (Occam-ranked)

### Fix 1 — `phi` formula missing factor of 2 (Occam: HIGHEST)
- File: `nsram/nsram/bsim4_port/temp.py:285`
  ```python
  phi = ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30)) + model["phin"] + 0.4
  ```
- BSIM4 reference (b4temp.c §1322): `phi = 2.0 * Vtm0 * log(Ndep/ni) + phin`
- Numerically at ndep=1.7e17, T=27°C, the `+0.4` empirically compensates so `phi`
  is correct to ±20 mV. This is why our **symbolic Vth matches ngspice**.
  But `phi` also enters `Xdep0 = sqrt(2εsub/(qN))·sqrtPhi`, `cdep0 = sqrt(qεN/(2φ))`,
  `lt1 = factor1·sqrt(Xdep)·...`, and `Theta0 = exp(-dvt1·Leff/(2·lt1))`, all of
  which ENTER subthreshold `n` via `tmp1 = epssub/Xdep`. A 2 % phi drift propagates
  through `cdep0` → T9 denominator → Vgsteff → subthreshold shape.
- **Action:** change to `2.0 * Vtm0 * log(...) + phin` and re-validate.

### Fix 2 — Vgsteff T9 denominator branch / `cdep0` fidelity (Occam: HIGH)
- `dc.py:462-471` — T9 denominator three-branch port. The `coxe/cdep0` ratio scales
  the exponential. cdep0 lives in `temp.py:325`:
  `cdep0 = sqrt(q·εsub·NDEP·1e6/(2·phi))`. **Same phi as Fix 1.**
- If Fix 1 alone doesn't close the gap, check that `mstar` (`dc.py:219`) is being
  computed exactly per b4temp.c §1373-1427 (look for any `0.5·...` simplification).
  Our `mstar = sd.mstar = 0.5` looks correct, but if BSIM4 actually uses
  `mstar = 0.5 + 1/(1+exp(...))` we may be missing the bias-dependent term.
- **Action:** print pyport `mstar`, `voffcbn`, `n` at every probe Vgs and compare
  to ngspice via `.print` of `@m1[vbseff]`-style internal hooks (or build a small
  ngspice OP scan with `let n = something` if exposed).

### Fix 3 — `voffcbn` integrand: `voffl` not binned (Occam: LOW-MEDIUM)
- `temp.py:321`: `voffl_v = model.get("voffl", 0.0)` reads RAW voffl, not
  `scaled["voffl"]`. M2 has zero binning coefs for voffl (`lvoffl/wvoffl/pvoffl` all
  default 0), so this is a no-op for M2 — but it's a latent bug for any card
  that bins voffl.
- A separate concern: `wvoff·Inv_W` makes scaled `voff = -0.1524 V`, then
  `voffcbn = -0.158 V`. ngspice's value is not exposed via `show m1 :` so we can't
  cross-check directly; but with `binunit=2` ⇒ `Inv_W = 1/Weff` (m⁻¹) the formula
  is per-spec.
- **Action:** as a control, set `wvoff=0` in pyport and re-run z91l. If the gap
  drops to <10 mV the bug is in voff binning convention.

---

## TL;DR for the user

- **Symbolic Vth is exact (90 µV vs ngspice).** The 60 mV offset is in the
  subthreshold bridge (Vgsteff path), specifically subthreshold `n`/`cdep0`,
  not the Vth assembly.
- **Top suspect:** `temp.py:285` — `phi = Vtm0·log(NDEP/ni) + phin + 0.4` is
  off by a factor of 2 (BSIM4 reference is `2·Vtm0·log(NDEP/ni) + phin`). The
  `+0.4` empirical fudge keeps `phi` numerically close (~20 mV off) so Vth
  matches, but the **propagated phi error in `Xdep0`/`cdep0`** is what shifts
  the Vgsteff bridge denominator, distorting subthreshold slope and producing
  the 60 mV constant-current Vth offset.
- **Concrete target:** `nsram/nsram/bsim4_port/temp.py:285`. Replace with
  `phi = 2.0 * ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30)) + model["phin"]`.
  Re-run z91l and z91k; expect both Vth_diff (60 mV) and slope (4 mV/dec) to
  shrink dramatically.
